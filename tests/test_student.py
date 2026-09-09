import pytest

torch = pytest.importorskip("torch")

from ospedit.student import HybridSpatialGraphStudent, ParentEditStudent, SpatialGraphStudent, encode_edit_features


def test_edit_encoder_marks_only_sequence_changes():
    features = encode_edit_features(["AAA"], ["AYA"])
    assert features.shape == (1, 3, 41)
    assert features[0, :, -1].tolist() == [0.0, 1.0, 0.0]


def test_student_has_exact_batch_identity_for_no_edit_samples():
    torch.manual_seed(0)
    model = ParentEditStudent(parent_dim=8, hidden_dim=32, blocks=2, heads=4)
    parent = torch.randn(2, 3, 8)
    edits = encode_edit_features(["AAA", "AAA"], ["AYA", "AAA"])
    output = model(parent, edits)
    assert output.shape == (2, 3, 6)
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    assert torch.isfinite(output).all()


def test_student_initializes_as_zero_update():
    model = ParentEditStudent(parent_dim=8, hidden_dim=32, blocks=2, heads=4)
    parent = torch.randn(1, 3, 8)
    edit = encode_edit_features(["AAA"], ["AYA"])
    assert torch.equal(model(parent, edit), torch.zeros(1, 3, 6))


def test_student_optional_delta_bound_is_respected():
    model = ParentEditStudent(parent_dim=8, hidden_dim=32, blocks=1, heads=4, max_normalized_delta=0.25)
    with torch.no_grad():
        model.output_projection[-1].bias.fill_(10.0)
    output = model(torch.randn(1, 2, 8), encode_edit_features(["AA"], ["AY"]))
    assert float(output.abs().max()) <= 0.25 + 1e-6


def test_student_positional_encoding_preserves_shape_and_is_optional():
    parent = torch.randn(1, 3, 8)
    edit = encode_edit_features(["AAA"], ["AYA"])
    encoded = ParentEditStudent(parent_dim=8, hidden_dim=32, blocks=1, heads=4)(parent, edit)
    plain = ParentEditStudent(parent_dim=8, hidden_dim=32, blocks=1, heads=4, use_positional_encoding=False)(parent, edit)
    assert encoded.shape == plain.shape == (1, 3, 6)


def test_student_padding_mask_blocks_padded_tokens():
    torch.manual_seed(2)
    model = ParentEditStudent(parent_dim=8, hidden_dim=32, blocks=2, heads=4)
    model.eval()
    parent = torch.randn(1, 3, 8)
    edits = encode_edit_features(["AAA"], ["AYA"])
    padded_parent = torch.cat((parent, torch.randn(1, 2, 8)), dim=1)
    padded_edits = torch.cat((edits, torch.zeros(1, 2, edits.shape[-1])), dim=1)
    short = model(parent, edits, residue_mask=torch.ones((1, 3), dtype=torch.bool))
    padded = model(padded_parent, padded_edits, residue_mask=torch.tensor([[True, True, True, False, False]]))
    assert torch.allclose(short, padded[:, :3], atol=1e-6)
    assert torch.equal(padded[:, 3:], torch.zeros_like(padded[:, 3:]))


def test_spatial_graph_student_requires_graph_and_preserves_no_edit_identity():
    model = SpatialGraphStudent(parent_dim=8, hidden_dim=16, blocks=2)
    parent = torch.randn(2, 3, 8)
    edits = encode_edit_features(["AAA", "AAA"], ["AYA", "AAA"])
    edge = torch.randn(2, 3, 3, 16)
    edge_mask = ~torch.eye(3, dtype=torch.bool)[None].expand(2, -1, -1)
    output = model(parent, edits, edge_features=edge, edge_mask=edge_mask)
    assert output.shape == (2, 3, 6)
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    with pytest.raises(ValueError, match="requires"):
        model(parent, edits)


def test_hybrid_spatial_graph_student_combines_graph_and_global_context():
    model = HybridSpatialGraphStudent(
        parent_dim=8, hidden_dim=16, graph_blocks=1, global_blocks=1, heads=4
    )
    parent = torch.randn(2, 3, 8)
    edits = encode_edit_features(["AAA", "AAA"], ["AYA", "AAA"])
    edge = torch.randn(2, 3, 3, 16)
    edge_mask = ~torch.eye(3, dtype=torch.bool)[None].expand(2, -1, -1)
    residue_mask = torch.ones((2, 3), dtype=torch.bool)
    output = model(
        parent,
        edits,
        residue_mask=residue_mask,
        edge_features=edge,
        edge_mask=edge_mask,
    )
    assert output.shape == (2, 3, 6)
    assert torch.equal(output[1], torch.zeros_like(output[1]))
