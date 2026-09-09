import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.endpoint_groups import endpoint_group_key
from scripts.augment_reverse_training import augment_reverse_training


def _record(pair_id="pair", split="train", reverse=False):
    source, target = ("C", "A") if reverse else ("A", "C")
    return PairRecord(
        pair=StructurePair(
            pair_id=pair_id,
            parent_sequence=f"A{source}",
            mutant_sequence=f"A{target}",
            parent_coords=np.zeros((2, 1, 3)) + (1 if reverse else 0),
            mutant_coords=np.zeros((2, 1, 3)) + (0 if reverse else 1),
            mutation_indices=(1,),
            atom_names=("CA",),
        ),
        parent_id="parent",
        family_id="family",
        split=split,
        source_file="target.pdb" if reverse else "source.pdb",
        target_file="source.pdb" if reverse else "target.pdb",
        source_chain="A",
        target_chain="A",
    )


def test_reverse_augmentation_swaps_endpoints_and_keeps_group():
    source = _record()

    output, report = augment_reverse_training([source])

    reverse = next(record for record in output if record.pair.pair_id.endswith("reverse_train"))
    assert reverse.pair.parent_sequence == source.pair.mutant_sequence
    assert np.array_equal(reverse.pair.parent_coords, source.pair.mutant_coords)
    assert endpoint_group_key(reverse) == endpoint_group_key(source)
    assert report["summary"]["directed_edit_types_after"] == 2


def test_reverse_augmentation_does_not_modify_dev_or_duplicate_existing_direction():
    forward = _record()
    reverse = _record("already_reverse", reverse=True)
    dev = _record("dev", split="dev")

    output, report = augment_reverse_training([forward, reverse, dev])

    assert len(output) == 3
    assert report["summary"]["reverse_train_records_added"] == 0
    assert report["summary"]["existing_reverse_records_skipped"] == 2
    assert report["summary"]["output_dev_records"] == 1
