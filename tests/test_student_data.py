import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair
from ospedit.student_data import PairDataset, collate_pair_records, iter_pair_batches, mutation_neighborhood_mask, parent_local_features, parent_residue_mask, parent_spatial_graph, target_local_delta


def make_record(pair_id="pair", length=2):
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    parent = np.repeat(residue[None, :, :], length, axis=0)
    mutant = parent.copy()
    mutant[1, :, 0] += 0.4
    pair = StructurePair(pair_id, "A" * length, "A" + "Y" + "A" * (length - 2), parent, mutant, (1,))
    return PairRecord(pair, "parent", "family", "dev")


def test_student_data_builds_local_features_and_target():
    record = make_record()
    features = parent_local_features(record.pair)
    target, valid = target_local_delta(record.pair)
    assert features.shape == (2, 16)
    assert target.shape == (2, 6)
    assert valid.tolist() == [True, True]
    assert np.isfinite(target).all()
    assert mutation_neighborhood_mask(record.pair).tolist() == [1.0, 1.0]


def test_pair_dataset_records_custom_neighborhood_radius():
    record = make_record()
    parent = record.pair.parent_coords.copy()
    parent[0, :, 1] -= 5.0
    pair = StructurePair("radius", "AA", "AY", parent, parent.copy(), (1,))
    item = PairDataset([PairRecord(pair, "parent", "family", "dev")], neighborhood_radius=0.1)[0]
    assert item["neighborhood_mask"].tolist() == [0.0, 1.0]


def test_parent_residue_mask_excludes_degenerate_frames():
    record = make_record()
    broken = record.pair.parent_coords.copy()
    broken[0] = np.nan
    pair = StructurePair("broken-parent", record.pair.parent_sequence, record.pair.mutant_sequence, broken, record.pair.mutant_coords, (1,), record.pair.atom_names)
    assert parent_residue_mask(pair).tolist() == [False, True]


def test_local_features_are_global_rigid_transform_invariant():
    record = make_record()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([4.0, 2.0, -1.0])
    transformed = record.pair.parent_coords @ rotation.T + shift
    transformed_pair = StructurePair(
        "transformed",
        record.pair.parent_sequence,
        record.pair.mutant_sequence,
        transformed,
        record.pair.mutant_coords @ rotation.T + shift,
        record.pair.mutation_indices,
        record.pair.atom_names,
    )
    assert np.allclose(parent_local_features(record.pair), parent_local_features(transformed_pair), atol=1e-6)


def test_target_delta_is_global_rigid_transform_invariant():
    record = make_record("target-invariant")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([2.0, -3.0, 4.0])
    def transform(coords):
        return coords @ rotation.T + shift
    transformed = StructurePair(
        "target-transformed",
        record.pair.parent_sequence,
        record.pair.mutant_sequence,
        transform(record.pair.parent_coords),
        transform(record.pair.mutant_coords),
        record.pair.mutation_indices,
        record.pair.atom_names,
    )
    original_target, original_mask = target_local_delta(record.pair)
    transformed_target, transformed_mask = target_local_delta(transformed)
    assert np.array_equal(original_mask, transformed_mask)
    assert np.allclose(original_target, transformed_target, atol=1e-6)


def test_collate_pads_variable_length_records():
    first = PairDataset([make_record("short", 2)])[0]
    second = PairDataset([make_record("long", 3)])[0]
    batch = collate_pair_records([first, second])
    assert batch["parent_features"].shape == (2, 3, 16)
    assert batch["target_delta"].shape == (2, 3, 6)
    assert batch["input_mask"].tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]


def test_pair_dataset_separates_parent_input_mask_from_target_loss_mask():
    record = make_record("mask-split")
    mutant = record.pair.mutant_coords.copy()
    mutant[0] = np.nan
    pair = StructurePair(
        record.pair.pair_id,
        record.pair.parent_sequence,
        record.pair.mutant_sequence,
        record.pair.parent_coords,
        mutant,
        record.pair.mutation_indices,
        record.pair.atom_names,
    )
    item = PairDataset([PairRecord(pair, "parent", "family", "dev")])[0]
    assert item["input_mask"].tolist() == [1.0, 1.0]
    assert item["loss_mask"].tolist() == [0.0, 1.0]


def test_pair_dataset_carries_scaled_teacher_delta_and_mask():
    record = make_record("teacher-delta")
    teacher = np.zeros((2, 6), dtype=np.float32)
    teacher[1, 0] = 2.0
    dataset = PairDataset(
        [record],
        translation_scale=2.0,
        rotation_scale=0.5,
        teacher_deltas={record.pair.pair_id: (teacher, np.array([True, False]))},
    )
    item = dataset[0]
    assert np.isclose(item["teacher_delta"][1, 0], 1.0)
    assert item["teacher_mask"].tolist() == [1.0, 0.0]
    batch = collate_pair_records([item])
    assert batch["teacher_delta"].shape == (1, 2, 6)
    assert batch["teacher_mask"].tolist() == [[1.0, 0.0]]


def test_pair_batches_have_reproducible_shuffle_order():
    dataset = PairDataset([make_record("a"), make_record("b"), make_record("c")])
    first = [batch["pair_ids"] for batch in iter_pair_batches(dataset, batch_size=1, shuffle=True, seed=7)]
    second = [batch["pair_ids"] for batch in iter_pair_batches(dataset, batch_size=1, shuffle=True, seed=7)]
    assert first == second


def test_pair_dataset_rejects_mixed_atom_layouts():
    first = make_record("first")
    coords = np.zeros((2, 1, 3))
    second_pair = StructurePair("second", "AA", "AY", coords, coords.copy(), (1,))
    with pytest.raises(ValueError, match="atom_names"):
        PairDataset([first, PairRecord(second_pair, "parent", "family", "dev")])


def test_pair_dataset_rejects_missing_oxygen_atom():
    coords = np.zeros((2, 3, 3))
    pair = StructurePair("missing-o", "AA", "AY", coords, coords.copy(), (1,), atom_names=("N", "CA", "C"))
    with pytest.raises(ValueError, match="N, CA, C, and O"):
        PairDataset([PairRecord(pair, "parent", "family", "dev")])


def test_masked_loss_ignores_padding():
    torch = pytest.importorskip("torch")
    from ospedit.student_training import masked_delta_loss

    prediction = torch.zeros((1, 2, 6))
    target = torch.zeros((1, 2, 6))
    target[0, 1] = 100.0
    mask = torch.tensor([[1.0, 0.0]])
    assert float(masked_delta_loss(prediction, target, mask)) == 0.0


def test_masked_prediction_norm_loss_ignores_padding():
    torch = pytest.importorskip("torch")
    from ospedit.student_training import masked_prediction_norm_loss

    prediction = torch.zeros((1, 2, 6))
    prediction[0, 1] = 10.0
    assert float(masked_prediction_norm_loss(prediction, torch.tensor([[1.0, 0.0]]))) == 0.0


def test_masked_delta_loss_averages_samples_not_residues():
    torch = pytest.importorskip("torch")
    from ospedit.student_training import masked_delta_loss

    prediction = torch.zeros((2, 3, 6))
    target = torch.ones_like(prediction)
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    # Both samples have unit per-residue MSE; length must not change the result.
    assert float(masked_delta_loss(prediction, target, mask)) == pytest.approx(1.0)


def test_masked_delta_loss_supports_smooth_l1_and_validates_options():
    torch = pytest.importorskip("torch")
    from ospedit.student_training import masked_delta_loss

    prediction = torch.zeros((1, 1, 6))
    target = torch.full_like(prediction, 2.0)
    assert float(masked_delta_loss(prediction, target, torch.ones((1, 1)), kind="smooth_l1", beta=1.0)) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="kind"):
        masked_delta_loss(prediction, target, torch.ones((1, 1)), kind="bad")


def test_masked_delta_loss_supports_residue_weights():
    torch = pytest.importorskip("torch")
    from ospedit.student_training import masked_delta_loss

    prediction = torch.zeros((1, 2, 6))
    target = torch.zeros_like(prediction)
    target[0, 0] = 2.0
    target[0, 1] = 1.0
    weighted = masked_delta_loss(prediction, target, torch.ones((1, 2)), residue_weights=torch.tensor([[3.0, 1.0]]))
    assert float(weighted) == pytest.approx(3.25)


def test_family_balanced_weights_survive_single_sample_batches():
    first = make_record("family-large-a")
    second = make_record("family-large-b")
    second = PairRecord(second.pair, second.parent_id, "large", second.split)
    first = PairRecord(first.pair, first.parent_id, "large", first.split)
    small = make_record("family-small")
    small = PairRecord(small.pair, small.parent_id, "small", small.split)
    dataset = PairDataset([first, second, small], family_balanced_loss=True)
    weights = [dataset[index]["sample_weight"] for index in range(3)]
    assert weights == pytest.approx([0.75, 0.75, 1.5])

    torch = pytest.importorskip("torch")
    from ospedit.student_training import masked_delta_loss

    prediction = torch.zeros((1, 1, 6))
    target = torch.ones_like(prediction)
    loss = masked_delta_loss(
        prediction,
        target,
        torch.ones((1, 1)),
        sample_weights=torch.tensor([1.5]),
    )
    assert float(loss) == pytest.approx(1.5)


def test_supervised_loss_normalizes_site_and_neighborhood_separately():
    torch = pytest.importorskip("torch")
    from ospedit.student_training import supervised_delta_loss

    prediction = torch.zeros((1, 4, 6))
    target = torch.zeros_like(prediction)
    target[:, 1] = 2.0
    edits = torch.zeros((1, 4, 41))
    edits[:, 1, -1] = 1.0
    neighborhood = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    loss = supervised_delta_loss(
        prediction,
        target,
        torch.ones((1, 4)),
        edits,
        neighborhood,
        mutation_weight=4.0,
        neighborhood_weight=1.0,
    )
    # Global MSE=1, mutation MSE=4, neighborhood MSE=2.
    assert float(loss) == pytest.approx(19.0)


def test_target_delta_scales_translation_and_rotation_channels():
    record = make_record()
    raw, mask = target_local_delta(record.pair)
    scaled, scaled_mask = target_local_delta(record.pair, translation_scale=2.0, rotation_scale=0.5)
    assert np.array_equal(mask, scaled_mask)
    assert np.allclose(scaled[..., :3], raw[..., :3] / 2.0)
    assert np.allclose(scaled[..., 3:], raw[..., 3:] / 0.5)


def test_target_delta_rejects_non_positive_scales():
    with pytest.raises(ValueError, match="positive"):
        target_local_delta(make_record().pair, translation_scale=0.0)


def test_parent_geometry_features_add_invariant_context_channels():
    record = make_record()
    base = parent_local_features(record.pair)
    enriched = parent_local_features(record.pair, include_geometry=True)
    assert enriched.shape == (record.pair.length, base.shape[-1] + 8)
    assert np.isfinite(enriched).all()


def test_parent_spatial_graph_is_global_rigid_transform_invariant():
    record = make_record("graph-invariant", length=3)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([3.0, -2.0, 1.0])
    transformed = StructurePair(
        "graph-transformed",
        record.pair.parent_sequence,
        record.pair.mutant_sequence,
        record.pair.parent_coords @ rotation.T + shift,
        record.pair.mutant_coords @ rotation.T + shift,
        record.pair.mutation_indices,
        record.pair.atom_names,
    )
    original_features, original_mask = parent_spatial_graph(record.pair, spatial_neighbors=1)
    transformed_features, transformed_mask = parent_spatial_graph(transformed, spatial_neighbors=1)
    assert np.array_equal(original_mask, transformed_mask)
    assert np.allclose(original_features, transformed_features, atol=1e-6)


def test_pair_dataset_collates_spatial_graph():
    item = PairDataset([make_record("graph", length=3)], include_spatial_graph=True, spatial_neighbors=1)[0]
    batch = collate_pair_records([item])
    assert batch["edge_features"].shape == (1, 3, 3, 16)
    assert batch["edge_mask"].shape == (1, 3, 3)
    assert not np.diag(batch["edge_mask"][0]).any()


def test_student_training_loop_runs_one_epoch():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_student

    batch = collate_pair_records([PairDataset([make_record()])[0]])
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_student(model, [batch], optimizer, epochs=1)
    assert len(history) == 1
    assert np.isfinite(history[0])


def test_spatial_graph_student_uses_common_training_loop():
    torch = pytest.importorskip("torch")
    from ospedit.student import SpatialGraphStudent
    from ospedit.student_training import train_student

    batch = collate_pair_records([
        PairDataset([make_record("graph-train", length=3)], include_spatial_graph=True)[0]
    ])
    model = SpatialGraphStudent(parent_dim=16, hidden_dim=16, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_student(model, [batch], optimizer, epochs=1)
    assert len(history) == 1 and np.isfinite(history[0])


def test_student_training_loop_supports_teacher_distillation():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_student

    record = make_record("distill")
    teacher = np.zeros((2, 6), dtype=np.float32)
    teacher[1, 0] = 0.25
    item = PairDataset([record], teacher_deltas={record.pair.pair_id: (teacher, np.ones(2, dtype=bool))})[0]
    batch = collate_pair_records([item])
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_student(model, [batch], optimizer, epochs=1, distill_weight=0.5)
    assert len(history) == 1 and np.isfinite(history[0])


def test_student_training_reuses_generator_across_epochs():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_student

    batch = collate_pair_records([PairDataset([make_record()])[0]])
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def batches():
        yield batch

    history = train_student(model, batches(), optimizer, epochs=2, gradient_clip_norm=0.5)
    assert len(history) == 2
    assert all(np.isfinite(value) for value in history)


def test_train_records_runs_from_pair_records():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_records

    records = [make_record("a"), make_record("b")]
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_records(model, records, optimizer, epochs=1, batch_size=2, shuffle=False)
    assert len(history) == 1
    assert np.isfinite(history[0])


def test_train_records_excludes_nonexperimental_labels_by_default():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_records

    record = PairRecord(make_record("teacher").pair, "parent", "family", "train", label_source="teacher")
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="experimental"):
        train_records(model, [record], optimizer, epochs=1)


def test_student_checkpoint_restores_model_and_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import load_student_checkpoint, save_student_checkpoint

    torch.manual_seed(4)
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reference = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "student.pt"
    save_student_checkpoint(model, path, optimizer=optimizer, epoch=3, history=[1.0, 0.5], config={"seed": 4})
    with torch.no_grad():
        for value in model.parameters():
            value.add_(10.0)
    metadata = load_student_checkpoint(model, path, optimizer=optimizer)
    assert metadata == {"epoch": 3, "history": [1.0, 0.5], "config": {"seed": 4}}
    assert all(torch.equal(value, reference[key]) for key, value in model.state_dict().items())


def test_student_training_gradient_accumulation_updates_after_window():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_student

    batch = collate_pair_records([PairDataset([make_record()])[0]])
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    train_student(model, [batch, batch], optimizer, epochs=1, grad_accumulation_steps=2)
    assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_gradient_accumulation_handles_partial_final_window():
    torch = pytest.importorskip("torch")
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import train_student

    batch = collate_pair_records([PairDataset([make_record()])[0]])
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_student(model, [batch, batch, batch], optimizer, epochs=1, grad_accumulation_steps=2)
    assert len(history) == 1 and np.isfinite(history[0])


def test_checkpoint_config_reports_architecture_mismatch():
    from ospedit.student_training import validate_student_checkpoint_config

    with pytest.raises(ValueError, match="hidden_dim"):
        validate_student_checkpoint_config({"hidden_dim": 128}, hidden_dim=256)
