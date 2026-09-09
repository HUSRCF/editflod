"""Deterministic, frame-restoring runner for the published PreMut checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np


BACKBONE_ATOMS = ("N", "CA", "C", "O")


def _backbone_indices(atom_names: list[str]) -> np.ndarray:
    residues: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for index, atom_name in enumerate(atom_names):
        if atom_name == "N":
            current = {}
            residues.append(current)
        if current is not None and atom_name in BACKBONE_ATOMS and atom_name not in current:
            current[atom_name] = index
    incomplete = [index for index, residue in enumerate(residues) if set(residue) != set(BACKBONE_ATOMS)]
    if incomplete:
        raise ValueError(f"PreMut atom stream has incomplete backbone residues: {incomplete[:5]}")
    return np.asarray([[residue[name] for name in BACKBONE_ATOMS] for residue in residues], dtype=int)


def _load_upstream(root: Path, checkpoint: Path, device: str):
    import torch

    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    previous = Path.cwd()
    os.chdir(root)
    try:
        from model import egnn_ablation_atom_types_only

        net = egnn_ablation_atom_types_only(
            in_channels=[3, 37, 21], hidden_channels=32, num_classes=3, num_hidden_layers=4
        )
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        remapped = {
            "model." + key.split("model.")[-1]: value
            for key, value in payload["state_dict"].items()
        }
        net.load_state_dict(remapped)
        net.to(device).eval()
        return net
    finally:
        os.chdir(previous)


def _seed(seed: int, device: str) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict_backbone(
    net,
    *,
    root: Path,
    parent: Path,
    chain: str,
    mutation: str,
    device: str,
    seed: int,
    parent_cache: dict[tuple[str, str], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    import torch
    from biopandas.pdb import PandasPdb
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.data import Data

    _seed(seed, device)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from constants import ATOMS
    from PDB_all_atom import PDBReader_All_Atom

    preprocess_started = perf_counter()
    cache_key = (str(parent), chain)
    cache_hit = parent_cache is not None and cache_key in parent_cache
    parse_started = perf_counter()
    if cache_hit:
        original_atoms = parent_cache[cache_key]
    else:
        pdb = PandasPdb().read_pdb(str(parent))
        atom_table = pdb.df["ATOM"]
        original_atoms = atom_table[
            (atom_table["chain_id"] == chain)
            & (atom_table["atom_name"].isin(ATOMS))
            & (atom_table["alt_loc"].isin(["A", ""]))
        ].copy()
        if parent_cache is not None:
            parent_cache[cache_key] = original_atoms
    parent_parse_seconds = perf_counter() - parse_started
    if original_atoms.empty:
        raise ValueError(f"PreMut parent contains no supported atoms for chain {chain!r}")
    centroid = original_atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float).mean(axis=0)

    normalized_id = parent.stem.lower()
    reader = PDBReader_All_Atom(
        pdb_dir=str(parent.parent),
        mutant_pdb=f"{normalized_id}_{chain}",
        wild_pdb=f"{normalized_id}_{chain}",
        mutation_info=mutation,
        state="predict",
    )
    # Upstream reparses the same PDB several times per graph. Its prediction
    # path only reads these frames, so serve the identity-filtered table above.
    def cached_pdb_df(*_args, **_kwargs):
        return original_atoms

    def cached_ppdb(*_args, **_kwargs):
        return None

    reader.get_pdb_df = cached_pdb_df
    reader.get_ppdb = cached_ppdb

    node_coords, _ = reader.get_xyz()
    node_atoms, node_residues, _ = reader.get_atoms()
    # Preserve the published predictor's RNG behavior: pdb_to_graph() calls
    # get_xyz() again while constructing KNN edges. Missing mutant atoms can
    # therefore receive a second random initialization for this edge query.
    edge_coords, _ = reader.get_xyz()
    neighbors = NearestNeighbors(n_neighbors=30).fit(edge_coords).kneighbors(edge_coords)
    edge_distances = reader.compute_edge_attr_from_neighbours_distance(torch.from_numpy(neighbors[0]))
    edge_index = reader.compute_edge_indexes_from_nearest_neighbours(torch.from_numpy(neighbors[1]))
    graph = Data(
        node_coords=torch.from_numpy(node_coords).float(),
        node_one_hot_sequence=node_atoms.float(),
        node_one_hot_sequence_residues=node_residues.float(),
        edge_index=edge_index,
        edge_distances=edge_distances.float(),
    )
    preprocess_seconds = perf_counter() - preprocess_started

    graph = graph.to(device)
    inference_started = perf_counter()
    with torch.inference_mode():
        displacement = net(
            node_feats=(graph.node_coords, graph.node_one_hot_sequence, graph.node_one_hot_sequence_residues),
            edge_index=graph.edge_index,
            edge_attr=graph.edge_distances,
            batch=getattr(graph, "batch", None),
        )
        predicted_atoms = displacement + graph.node_coords
    inference_seconds = perf_counter() - inference_started

    postprocess_started = perf_counter()
    atom_type_indices = graph.node_one_hot_sequence.argmax(dim=-1).detach().cpu().numpy()
    atom_names = [ATOMS[int(index)] for index in atom_type_indices]
    backbone_indices = _backbone_indices(atom_names)
    input_backbone = graph.node_coords.detach().cpu().numpy()[backbone_indices] + centroid
    prediction = predicted_atoms.detach().cpu().numpy()[backbone_indices] + centroid
    postprocess_seconds = perf_counter() - postprocess_started
    return prediction, input_backbone, {
        "preprocess_seconds": preprocess_seconds,
        "parent_parse_seconds": parent_parse_seconds,
        "parent_cache_hit": float(cache_hit),
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic PreMut inference")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-structure", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--mutation", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--source-checksum", required=True)
    parser.add_argument("--runner-checksum", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    began = perf_counter()
    root = Path(args.upstream_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    parent = Path(args.parent_structure).expanduser().resolve()
    if not (root / "src" / "PDB_all_atom.py").is_file():
        raise SystemExit(f"invalid PreMut root: {root}")
    if not checkpoint.is_file() or not parent.is_file():
        raise SystemExit("checkpoint and parent structure must be files")

    _seed(args.seed, args.device)
    net = _load_upstream(root, checkpoint, args.device)
    model_ready = perf_counter()
    prediction, input_backbone, runtime = predict_backbone(
        net,
        root=root,
        parent=parent,
        chain=args.chain,
        mutation=args.mutation,
        device=args.device,
        seed=args.seed,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "ospedit.premut_prediction.v1",
        "pair_id": args.pair_id,
        "source_checksum": args.source_checksum,
        "checkpoint_checksum": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "runner_checksum": args.runner_checksum,
        "upstream_revision": None if args.upstream_revision == "unknown" else args.upstream_revision,
        "device": args.device,
        "seed": args.seed,
        "mutation": args.mutation,
        "chain": args.chain,
        "runtime": {
            "model_load_seconds": model_ready - began,
            **runtime,
        },
    }
    np.savez_compressed(
        output,
        prediction=prediction,
        input_backbone=input_backbone,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


if __name__ == "__main__":
    main()
