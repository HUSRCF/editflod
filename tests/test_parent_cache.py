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


def test_geometry_cache_reuses_parent_context_but_recomputes_edit_channels():
    residue = np.asarray(
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    )
    coords = np.repeat(residue[None], 3, axis=0)
    coords[:, :, 1] += np.arange(3)[:, None] * 4.0
    first = StructurePair("first", "AAA", "YAA", coords, coords.copy(), (0,))
    second = StructurePair("second", "AAA", "AAY", coords.copy(), coords.copy(), (2,))
    cache = ParentContextCache(include_geometry=True)

    first_features = cache.get(first)
    second_features = cache.get(second)

    assert cache.misses == 1 and cache.hits == 1
    assert first_features[0, -1] == 1.0 and first_features[2, -1] == 0.0
    assert second_features[0, -1] == 0.0 and second_features[2, -1] == 1.0
    assert not np.array_equal(first_features[:, -2:], second_features[:, -2:])
