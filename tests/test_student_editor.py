import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ospedit.data import StructurePair
from ospedit.experiment import evaluate_editor
from ospedit.models import StudentEditor
from ospedit.student_inference import ParentContextCache


class ZeroStudent(torch.nn.Module):
    def forward(self, parent, edit):
        return torch.zeros((parent.shape[0], parent.shape[1], 6), dtype=parent.dtype, device=parent.device)


def test_student_editor_uses_common_runtime_accounting():
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("student-editor", "AA", "AY", coords, coords.copy(), (1,))
    result = evaluate_editor(pair, StudentEditor(ZeroStudent()))
    assert result.runtime.network_calls == 1
    assert result.runtime.sequence_encoder_calls == 0
    assert result.runtime.structure_updates == 1


def test_student_editor_reports_parent_cache_counters():
    coords = np.array(
        [[[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
         [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]]]
    )
    pair = StructurePair("student-cache", "AA", "AY", coords, coords.copy(), (1,))
    cache = ParentContextCache()
    result = evaluate_editor(pair, StudentEditor(ZeroStudent(), parent_cache=cache))
    assert result.runtime.extras["parent_cache_misses"] == 1.0
    assert result.runtime.extras["parent_cache_entries"] == 1.0


def test_parent_context_cache_returns_read_only_shared_features():
    coords = np.array(
        [[[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
         [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]]]
    )
    pair = StructurePair("readonly-cache", "AA", "AY", coords, coords.copy(), (1,))
    cache = ParentContextCache()
    features = cache.get(pair)
    assert features.flags.writeable is False
    assert cache.get(pair) is features
    with pytest.raises(ValueError, match="read-only"):
        features[0, 0] = 1.0


def test_student_editor_rejects_cache_geometry_mismatch():
    with pytest.raises(ValueError, match="geometry"):
        StudentEditor(ZeroStudent(), parent_cache=ParentContextCache(), include_geometry=True)
