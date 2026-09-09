import numpy as np

from ospedit.data import StructurePair
from ospedit.student_inference import ParentContextCache


def test_parent_context_cache_reuses_identical_parent_features():
    coords = np.asarray([
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    ])
    first = StructurePair("a", "A", "Y", coords, coords, (0,), ("N", "CA", "C", "O"))
    second = StructurePair("b", "A", "G", coords.copy(), coords, (0,), ("N", "CA", "C", "O"))
    cache = ParentContextCache()
    left = cache.get(first)
    right = cache.get(second)
    assert left is right
    assert len(cache) == 1
    assert cache.misses == 1
    assert cache.hits == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_parent_context_cache_can_bound_memory_with_lru_eviction():
    coords = np.asarray([
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    ])
    first = StructurePair("a", "A", "Y", coords, coords, (0,), ("N", "CA", "C", "O"))
    second_coords = coords.copy()
    second_coords[0, 1, 0] = 0.1
    second = StructurePair("b", "A", "G", second_coords, second_coords, (0,), ("N", "CA", "C", "O"))
    cache = ParentContextCache(max_entries=1)
    cache.get(first)
    cache.get(second)
    assert len(cache) == 1
    cache.get(first)
    assert cache.misses == 3
