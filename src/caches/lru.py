import torch  # PyTorch deep learning library
from torch.utils.data import TensorDataset
import torch.nn.functional as F  # PyTorch functional interface

from src.caches.base import BaseCache, register_class
from typing import override, Optional, List


@register_class("lru")
class LRUCache(BaseCache):
    """
    LRU cache with redundancy-aware eviction.
    If capacity is reached, evicts the least-recently used vector; ties are
    broken by evicting the vector that is most similar to any other vector.

    Parameters
    ----------
    capacity
        The maximum size of the cache.
    encodings
        The encodings of the training data.
    labels
        The labels of the training data.
    d_thresh
        The distance threshold for the smart cache.
    k
        The number of nearest neighbors to consider for the smart cache.
    """
    def __init__(
        self,
        encodings: Optional[torch.Tensor] = None,
        labels: Optional[List[int]] = None,
        d_thresh: float = 0.5,
        k: int = 5,
        capacity: int = 100,
    ) -> None:
        super().__init__(encodings=None, labels=None, d_thresh=d_thresh, k=k)
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self._capacity = capacity
        # Track access timestamps for LRU eviction
        self._access_time = torch.zeros(0, dtype=torch.long)
        self._time_counter = 0
        if encodings is not None and labels is not None:
            self.fit(encodings, labels)


    @override
    def top_k(self, query: torch.Tensor, update_counter: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute the Cosine distance between the query and the database vectors
        dist = 1 - F.cosine_similarity(query.to(self.database.device), self.database, dim=1)
        # Get the top k nearest neighbors
        d, top_k = torch.topk(dist, k=min(self.k, len(dist)), dim=0, largest=False)
        
        # LRU update: update access times for accessed items
        if update_counter:
            self._time_counter += 1
            self._access_time[top_k] = self._time_counter
        
        # Reshape the tensors to have the same shape
        top_k , d = top_k.unsqueeze(0), d.unsqueeze(0)
        return d, top_k
    

    def _topk_w_centroid(self, query: str | torch.Tensor) -> torch.Tensor:
        """
        Get the weighted centroid of the top k nearest neighbors of the query.

        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        torch.Tensor
            The weighted centroid of the top k nearest neighbors of the query.
        """
        # Get the top k nearest neighbors
        d, top_k = self.top_k(query, update_counter=False)
        # Add weights to the top k nearest neighbors
        weights = 1 / (d + 1e-6) ** 2
        # normalize the weights
        weights = weights / torch.sum(weights)
        # Compute the weighted centroid by multiplying the weights with the top k nearest neighbors
        weighted_centroid = weights @ self.database[top_k]
        return weighted_centroid.squeeze(0)
    

    @override
    def is_near(self, query: torch.Tensor) -> bool:
        """
        Check if the query is near the weighted centroid of the top k nearest neighbors.
        
        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        bool
            True if the query is near the weighted centroid of the top k nearest neighbors based on the
            distance threshold, False otherwise.
        """
        if len(self) == 0:
            return False
        # Calculate the weighted centroid
        weighted_centroid = self._topk_w_centroid(query)
        # Calculate the distance between the query and the weighted centroid
        dist = 1 - F.cosine_similarity(query.to(device=weighted_centroid.device), weighted_centroid, dim=1)
        return torch.any(dist < self.d_thresh).item()



    @override
    def add(self, query: torch.Tensor, label: int | torch.Tensor) -> None:
        if isinstance(label, torch.Tensor):
            label = label.item()
            assert isinstance(label, int)

        # If similar enough to existing vectors and we want deduplication,
        # simply skip insertion.
        if self.is_near(query):
            return
        
        # CRITICAL FIX: Check capacity and evict if necessary
        if len(self) < self._capacity:
            # append
            self.database = torch.cat((self.database, query.unsqueeze(0).to(self.database.device)))
            self.labels.append(label)
            self._time_counter += 1
            self._access_time = torch.cat([self._access_time, torch.tensor([self._time_counter], dtype=torch.long, device=self._access_time.device)])
            return

        # --------  Eviction path  -------- #
        # CRITICAL FIX: This is where we enforce capacity
        # 1. Find least recently used items
        lru_time = torch.min(self._access_time)
        candidate_idx = torch.nonzero(self._access_time == lru_time, as_tuple=False).squeeze(1)

        # 2. redundancy score: similarity to nearest other vector
        #    (we use 1 - cosine similarity as distance)
        if len(candidate_idx) > 1:
            sim_to_rest = []
            for i in candidate_idx.tolist():
                # similarity to all others except itself
                candidate_v =  self.database[i]
                sim = F.cosine_similarity(candidate_v.unsqueeze(0), self.database, dim=1)
                sim[i] = -1.0  # ignore self-similarity
                sim_to_rest.append(torch.max(sim))
            sim_to_rest = torch.stack(sim_to_rest)
            # pick the most redundant (highest similarity)
            evict_pos = candidate_idx[torch.argmax(sim_to_rest)]
        else:
            evict_pos = candidate_idx[0] if candidate_idx.numel() > 0 else candidate_idx.item()

        # 3. in-place replacement keeps tensor contiguous
        self.database[evict_pos] = query
        self.labels[evict_pos] = label
        self._time_counter += 1
        self._access_time[evict_pos] = self._time_counter  # reset access time

    
    @override
    def fit(self, vectors: torch.Tensor, labels: list[int]) -> None:
        """
        Fit the cache with initial vectors and labels.
        CRITICAL FIX: Respect capacity during initialization.
        """
        # Only add up to capacity items
        if len(vectors) != len(labels):
            raise ValueError("Vectors and labels must have the same length.")
        n_items = min(len(vectors), self._capacity)
        for i in range(n_items):
            self.add(vectors[i], labels[i])
    
    
    def get_last_p_added(self, p: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the last p items that were most recently added/accessed.
        
        Parameters
        ----------
        p
            Number of items to retrieve.
            
        Returns
        -------
        List[tuple[torch.Tensor, torch.Tensor]]
            List of (embedding, label) tuples.
        """
        if len(self) == 0:
            return []
        
        # Get indices of p most recently accessed items
        p = min(p, len(self))
        _, recent_idx = torch.topk(self._access_time, k=p, largest=True)
        
        items = []
        for idx in recent_idx:
            # Ensure proper shape for DataLoader compatibility
            embedding = self.database[idx].unsqueeze(0)
            label_tensor = torch.tensor([self.labels[idx]], dtype=torch.long, device=self.database.device)
            items.append((embedding, label_tensor))
            
        return items