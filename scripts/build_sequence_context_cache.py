"""Precompute frozen per-residue protein language-model embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ospedit.data import load_manifest
from ospedit.sequence_context import write_sequence_context_cache


def encode_sequences(
    sequences: list[str],
    *,
    model_id: str,
    revision: str | None,
    batch_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], str | None]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional experiment dependency.
        raise RuntimeError(
            "sequence context generation requires torch and transformers"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded: dict[str, Any] = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask").to(dtype=torch.bool)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state.detach().cpu()
        attention = encoded["attention_mask"].detach().cpu().to(dtype=torch.bool)
        for index, sequence in enumerate(batch):
            residue_mask = attention[index] & ~special[index]
            values = hidden[index, residue_mask].to(dtype=torch.float32).numpy()
            if values.shape[0] != len(sequence):
                raise ValueError(
                    f"tokenizer produced {values.shape[0]} residue tokens for "
                    f"sequence length {len(sequence)}"
                )
            embeddings[sequence] = values
    return embeddings, resolved_revision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen sequence-context cache for an ospedit manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    records = load_manifest(args.manifest)
    sequences = sorted(
        {
            sequence
            for record in records
            for sequence in (record.pair.parent_sequence, record.pair.mutant_sequence)
        }
    )
    embeddings, revision = encode_sequences(
        sequences,
        model_id=args.model_id,
        revision=args.revision,
        batch_size=args.batch_size,
        device=args.device,
    )
    metadata = write_sequence_context_cache(
        args.output,
        embeddings,
        model_id=args.model_id,
        model_revision=revision,
    )
    print(
        f"sequence context cache written: {Path(args.output).resolve()} "
        f"({metadata['sequence_count']} sequences, dim={metadata['embedding_dim']})"
    )


if __name__ == "__main__":
    main()
