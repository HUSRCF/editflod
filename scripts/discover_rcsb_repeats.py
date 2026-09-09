"""Discover context-matched same-sequence repeat structures through RCSB."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from Bio.PDB import MMCIFParser, PDBParser

from ospedit.data import PairRecord, load_manifest, manifest_fingerprint, parse_structure
from scripts.audit_structure_context import _structure_context


SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.{suffix}"
PDB_ENTITY_PATTERN = re.compile(r"^([A-Za-z0-9]{4})_\d+$")
REPEAT_COLUMNS = (
    "pair_id",
    "parent_structure",
    "parent_chain",
    "repeat_structure",
    "repeat_chain",
    "mutation_index",
)


def rcsb_sequence_entries(
    sequence: str,
    *,
    max_results: int = 100,
    timeout: float = 30.0,
    retries: int = 2,
) -> list[str]:
    """Return experimental PDB entry IDs from an exact-identity sequence query."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    if max_results <= 0 or timeout <= 0 or retries < 0:
        raise ValueError("max_results/timeout must be positive and retries non-negative")
    payload = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": 1,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_verbosity": "minimal",
        },
        "return_type": "polymer_entity",
    }
    request = Request(
        SEARCH_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ospedit/0.1"},
    )
    result = None
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                result = json.load(response)
            last_error = None
            break
        except (HTTPError, URLError, OSError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    if result is None:
        assert last_error is not None
        raise last_error
    entries = set()
    for item in result.get("result_set", []):
        match = PDB_ENTITY_PATTERN.fullmatch(str(item.get("identifier", "")))
        if match is not None:
            entries.add(match.group(1).upper())
    return sorted(entries)


def download_rcsb_entries(
    entry_ids: Iterable[str],
    destination: str | Path,
    *,
    timeout: float = 30.0,
    retries: int = 2,
    max_workers: int = 8,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Download PDB files with a persistent per-entry cache."""
    if timeout <= 0 or retries < 0 or max_workers <= 0:
        raise ValueError("timeout/workers must be positive and retries non-negative")
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identifiers = sorted(set(entry_ids))
    for entry_id in identifiers:
        if re.fullmatch(r"[A-Za-z0-9]{4}", entry_id) is None:
            raise ValueError(f"invalid PDB entry ID: {entry_id!r}")

    def download_one(entry_id: str) -> tuple[Path | None, dict[str, str] | None]:
        for suffix in ("pdb", "cif"):
            cached = root / f"{entry_id.upper()}.{suffix}"
            if cached.is_file() and cached.stat().st_size:
                return cached, None
        last_error: Exception | None = None
        for suffix in ("pdb", "cif"):
            target = root / f"{entry_id.upper()}.{suffix}"
            temporary = target.with_suffix(f".{suffix}.tmp")
            for attempt in range(retries + 1):
                try:
                    request = Request(
                        DOWNLOAD_TEMPLATE.format(pdb_id=entry_id.upper(), suffix=suffix),
                        headers={"User-Agent": "ospedit/0.1"},
                    )
                    with urlopen(request, timeout=timeout) as response:  # noqa: S310
                        content = response.read()
                    if not content:
                        raise OSError("empty response")
                    temporary.write_bytes(content)
                    temporary.replace(target)
                    return target, None
                except (HTTPError, URLError, OSError) as error:
                    last_error = error
                    target.unlink(missing_ok=True)
                    temporary.unlink(missing_ok=True)
                    if attempt < retries:
                        time.sleep(0.25 * (attempt + 1))
        assert last_error is not None
        return None, {"pdb_id": entry_id.upper(), "reason": str(last_error)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_one, identifiers))
    paths = [path for path, _ in results if path is not None]
    failures = [failure for _, failure in results if failure is not None]
    return paths, failures


def discover_repeat_rows(
    records: Iterable[PairRecord],
    candidate_paths: Iterable[str | Path],
    *,
    ignored_hetero: Iterable[str] = ("HOH",),
    require_single_protein_chain: bool = True,
    require_matching_target_hetero: bool = True,
    require_matching_method: bool = True,
) -> tuple[list[dict[str, str | int]], dict[str, int]]:
    """Filter downloaded candidates using observed sequence and context."""
    rows = list(records)
    ignored = frozenset(value.strip().upper() for value in ignored_hetero if value.strip())
    sequences = {record.pair.parent_sequence for record in rows}
    parsed_candidates: list[tuple[Path, str, Any, dict[str, Any]]] = []
    counters: Counter[str] = Counter()
    for raw_path in sorted({Path(path).expanduser().resolve() for path in candidate_paths}):
        try:
            parser = (
                MMCIFParser(QUIET=True)
                if raw_path.suffix.lower() in {".cif", ".mmcif"}
                else PDBParser(QUIET=True, PERMISSIVE=True)
            )
            structure = parser.get_structure(raw_path.stem, str(raw_path))
            model = next(iter(structure))
        except (OSError, StopIteration, ValueError):
            counters["unreadable_candidate_files"] += 1
            continue
        for chain in model:
            counters["candidate_chains"] += 1
            try:
                parsed = parse_structure(raw_path, chain.id)
                context = _structure_context(raw_path, chain.id, ignored_hetero=ignored)
            except (OSError, ValueError):
                counters["unreadable_candidate_chains"] += 1
                continue
            if parsed.sequence in sequences:
                parsed_candidates.append((raw_path, chain.id, parsed, context))
            else:
                counters["observed_sequence_mismatch"] += 1

    output: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in rows:
        source_path = Path(record.source_file).expanduser().resolve()
        source_entry = source_path.stem.upper()
        try:
            source_parsed = parse_structure(source_path, record.source_chain)
            parent_context = _structure_context(
                source_path,
                record.source_chain,
                ignored_hetero=ignored,
            )
        except (OSError, ValueError):
            counters["unreadable_parent"] += 1
            continue
        if source_parsed.sequence != record.pair.parent_sequence:
            counters["cropped_or_nonmatching_parent"] += 1
            continue
        for path, chain_id, parsed, context in parsed_candidates:
            if parsed.sequence != record.pair.parent_sequence:
                continue
            if path.stem.upper() == source_entry and chain_id == record.source_chain:
                continue
            if require_single_protein_chain and context["protein_chain_count"] != 1:
                counters["multiple_protein_chains"] += 1
                continue
            if (
                require_matching_target_hetero
                and context["target_hetero"] != parent_context["target_hetero"]
            ):
                counters["target_hetero_mismatch"] += 1
                continue
            if require_matching_method and (
                context["structure_method"] != parent_context["structure_method"]
            ):
                counters["structure_method_mismatch"] += 1
                continue
            key = (record.pair.pair_id, str(path), chain_id)
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "pair_id": record.pair.pair_id,
                "parent_structure": str(source_path),
                "parent_chain": record.source_chain,
                "repeat_structure": str(path),
                "repeat_chain": chain_id,
                "mutation_index": record.pair.mutation_indices[0],
            })
    counters["repeat_rows"] = len(output)
    counters["pairs_with_repeats"] = len({str(row["pair_id"]) for row in output})
    return output, dict(sorted(counters.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover same-sequence RCSB repeat structures")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--pairs-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--query-cache")
    parser.add_argument("--ignore-hetero", nargs="*", default=("HOH",))
    parser.add_argument("--allow-multiple-protein-chains", action="store_true")
    parser.add_argument("--allow-target-hetero-mismatch", action="store_true")
    parser.add_argument("--allow-method-mismatch", action="store_true")
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    query_cache_path = Path(args.query_cache) if args.query_cache else None
    query_failures: list[dict[str, Any]] = []
    requested_sequences = sorted({record.pair.parent_sequence for record in records})
    try:
        cached_query_results: dict[str, list[str]] = (
            json.loads(query_cache_path.read_text())
            if query_cache_path is not None and query_cache_path.is_file()
            else {}
        )
        if not isinstance(cached_query_results, dict) or any(
            not isinstance(key, str) or not isinstance(value, list)
            for key, value in cached_query_results.items()
        ):
            raise ValueError("query cache must map sequence strings to entry-ID lists")
        query_results = {
            sequence: cached_query_results[sequence]
            for sequence in requested_sequences
            if sequence in cached_query_results
        }
        for sequence in requested_sequences:
            if sequence in query_results:
                continue
            try:
                query_results[sequence] = rcsb_sequence_entries(
                    sequence,
                    max_results=args.max_results,
                    timeout=args.timeout,
                    retries=args.retries,
                )
            except (HTTPError, URLError, OSError, ValueError) as error:
                query_failures.append({
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "sequence_length": len(sequence),
                    "reason": str(error),
                })
                continue
            if query_cache_path is not None:
                query_cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = query_cache_path.with_suffix(query_cache_path.suffix + ".tmp")
                cached_query_results.update(query_results)
                temporary.write_text(
                    json.dumps(cached_query_results, indent=2, sort_keys=True) + "\n"
                )
                temporary.replace(query_cache_path)
        entry_ids = sorted({entry for entries in query_results.values() for entry in entries})
        paths, download_failures = download_rcsb_entries(
            entry_ids,
            args.download_dir,
            timeout=args.timeout,
            retries=args.retries,
            max_workers=args.download_workers,
        )
        rows, counters = discover_repeat_rows(
            records,
            paths,
            ignored_hetero=args.ignore_hetero,
            require_single_protein_chain=not args.allow_multiple_protein_chains,
            require_matching_target_hetero=not args.allow_target_hetero_mismatch,
            require_matching_method=not args.allow_method_mismatch,
        )
    except (HTTPError, URLError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.pairs_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPEAT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "format": "ospedit.rcsb_repeat_discovery.v1",
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "manifest_fingerprint": manifest_fingerprint(records),
        "configuration": {
            "max_results_per_sequence": args.max_results,
            "download_workers": args.download_workers,
            "query_cache": str(query_cache_path.resolve()) if query_cache_path else None,
            "ignored_hetero": sorted({value.strip().upper() for value in args.ignore_hetero}),
            "require_single_protein_chain": not args.allow_multiple_protein_chains,
            "require_matching_target_hetero": not args.allow_target_hetero_mismatch,
            "require_matching_method": not args.allow_method_mismatch,
        },
        "unique_sequences": len(query_results),
        "requested_unique_sequences": len(requested_sequences),
        "query_failures": query_failures,
        "unique_entry_ids": len(entry_ids),
        "downloaded_or_cached_files": len(paths),
        "download_failures": download_failures,
        "counters": counters,
        "repeat_counts": dict(sorted(Counter(str(row["pair_id"]) for row in rows).items())),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"RCSB repeat discovery written: {destination} "
        f"({len(rows)} rows for {counters.get('pairs_with_repeats', 0)} pairs)"
    )


if __name__ == "__main__":
    main()
