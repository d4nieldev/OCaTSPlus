import pytest
import torch
from src.caches.lru import LRUCache

def make_orthogonal_vector(idx, dim=768):
    # Create an orthogonal basis vector with 1.0 at index `idx`
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v

# Test that initializing with non-positive capacity raises ValueError
def test_init_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
    with pytest.raises(ValueError):
        LRUCache(capacity=-1)

# Test adding vectors respects capacity, length updates correctly,
# and eviction of least recently used happens when full
def test_add_and_len():
    cache = LRUCache(capacity=3, d_thresh=0.5)  # reasonable threshold
    v1 = make_orthogonal_vector(0)
    v2 = make_orthogonal_vector(1)
    v3 = make_orthogonal_vector(2)
    v4 = make_orthogonal_vector(3)

    cache.add(v1, 10)
    assert len(cache) == 1
    cache.add(v2, 20)
    cache.add(v3, 30)
    assert len(cache) == 3

    # Adding when full evicts the least recently used entry
    cache.add(v4, 40)
    assert len(cache) == 3
    assert 40 in cache.labels
    assert 10 not in cache.labels

# Test that adding a vector near an existing one does not increase cache size (duplicate skip)
def test_add_duplicate_skip():
    cache = LRUCache(capacity=5, d_thresh=0.5)
    v1 = make_orthogonal_vector(1)
    cache.add(v1, 1)
    length_before = len(cache)
    cache.add(v1 * 1.000001, 2)  # Slightly different vector, should be skipped as duplicate
    assert len(cache) == length_before
    assert 2 not in cache.labels

# Test that is_near returns True for vectors close to the cache and False for distant vectors
def test_is_near():
    cache = LRUCache(capacity=5, d_thresh=0.5)
    v1 = make_orthogonal_vector(1)
    cache.add(v1, 1)
    assert cache.is_near(v1)
    v_far = torch.ones_like(v1) * 10  # Far away vector
    assert not cache.is_near(v_far)

# Test that calling top_k updates the usage timestamp of the returned index
def test_top_k_updates_timestamp():
    cache = LRUCache(capacity=3, d_thresh=0.5)
    v1 = make_orthogonal_vector(1)
    v2 = make_orthogonal_vector(2)
    cache.add(v1, 1)
    cache.add(v2, 2)

    counter_before = cache._time_counter
    dist, idx = cache.top_k(v1)
    assert dist.shape == (1, 2)
    assert idx.shape == (1, 2)
    assert cache._time_counter == counter_before + 1

# Test that fit properly resets the cache database, labels, timestamps, and counter
def test_fit_resets_cache():
    cache = LRUCache(capacity=10)
    vectors = torch.stack([make_orthogonal_vector(i) for i in range(3)])  # Shape (3, dim)
    labels = [10, 20, 30]
    cache.fit(vectors, labels)
    assert len(cache) == 3
    assert cache.labels == labels
    assert cache._time_counter == 3
    assert torch.equal(cache._access_time, torch.tensor([1, 2, 3]))

# Test that fit raises a ValueError if vectors and labels have mismatched lengths
def test_fit_raises_on_mismatch():
    cache = LRUCache()
    vectors = torch.stack([make_orthogonal_vector(i) for i in range(2)])
    labels = [1]
    with pytest.raises(ValueError):
        cache.fit(vectors, labels)
