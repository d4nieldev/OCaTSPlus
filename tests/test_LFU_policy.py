import torch
import pytest
from src.caches.lfu import LFUCache
from src.caches.lru import LRUCache


def make_data(n=10, dim=768):
    # Generate normalized random vectors and sequential labels
    data = torch.randn(n, dim)
    data = torch.nn.functional.normalize(data, dim=1)
    labels = list(range(n))
    return data, labels


class TestLFUCache:

    # Test initialization and fitting: cache database, labels, and frequency initialization
    def test_init_and_fit(self):
        data, labels = make_data(5, 768)
        cache = LFUCache(encodings=data, labels=labels, capacity=10)
        db_shape = cache.database.shape
        if len(db_shape) == 1:
            actual_len = 1
        else:
            actual_len = db_shape[0]
        assert actual_len == 5

        assert cache.labels == labels
        assert torch.all(cache._freq == 1)

    # Test adding vectors when cache is under capacity: database and frequency sizes update correctly
    def test_add_under_capacity(self):
        data, labels = make_data(3, 768)
        cache = LFUCache(capacity=5)
        for v, l in zip(data, labels):
            cache.add(v, l)
        db_shape = cache.database.shape
        if len(db_shape) == 1:
            actual_len = 1
        else:
            actual_len = db_shape[0]
        assert actual_len == 3
        assert cache.labels == labels
        assert cache._freq.shape[0] == 3

    # Test adding vectors when cache reaches capacity triggers eviction and frequency updates
    def test_add_with_eviction(self):
        capacity = 3
        data, labels = make_data(5, 768)
        cache = LFUCache(capacity=capacity)
        # Add initial capacity vectors
        for i in range(capacity):
            cache.add(data[i], labels[i])
        assert len(cache) == capacity

        # Add more vectors to force eviction
        for i in range(capacity, len(data)):
            cache.add(data[i], labels[i])

        # Cache size remains at capacity after eviction
        assert len(cache) == capacity
        # Frequencies vector length matches capacity
        assert cache._freq.shape[0] == capacity

    # Test top_k returns expected shape and updates frequency counts for returned indices
    def test_top_k_and_frequency_update(self):
        data, labels = make_data(6, 768)
        cache = LFUCache(encodings=data, labels=labels, capacity=10)
        query = data[0]  # 1D query for LFU
        dist, topk = cache.top_k(query)
        # Check shapes: topk and dist should be 2D with shape (1, k)
        assert dist.dim() == 2 and topk.dim() == 2
        # Frequencies of returned topk indices should have increased (>1)
        indices = topk.squeeze(0)
        for idx in indices:
            assert cache._freq[idx] > 1

    # Test is_near returns True for identical vector and returns bool for unrelated vector
    def test_is_near_true_and_false(self):
        data, labels = make_data(4, 768)
        cache = LFUCache(encodings=data, labels=labels, d_thresh=0.6, capacity=10)
        query = data[0]  # 1D query for LFU
        # Identical vector should be near (True)
        assert cache.is_near(query).item() == True

        # Random normalized vector might be near or far — just sanity check
        unrelated = torch.randn(768)
        unrelated = unrelated / unrelated.norm()
        assert cache.is_near(unrelated).item() in [True, False]

    # Test is_near returns False on an empty cache (no vectors stored)
    def test_is_near_empty_cache(self):
        cache = LFUCache(capacity=5)
        query = torch.randn(768)  # 1D query for LFU
        assert cache.is_near(query) == False



# Helper to compute the number of vectors in a cache's database, handling edge cases
def db_len(cache):
    shape = cache.database.shape
    if len(shape) == 1:
        return 1
    return shape[0]
