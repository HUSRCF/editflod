"""Normalized RCSB entry and biological-assembly metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"
CACHE_FORMAT = "ospedit.rcsb_environment_metadata.v2"
LEGACY_CACHE_FORMATS = {"ospedit.rcsb_environment_metadata.v1"}
ENTRY_QUERY = """
query($ids:[String!]!) {
  entries(entry_ids:$ids) {
    rcsb_id
    struct { title }
    rcsb_primary_citation { title year pdbx_database_id_DOI pdbx_database_id_PubMed }
    rcsb_entry_info { experimental_method resolution_combined }
    cell { length_a length_b length_c angle_alpha angle_beta angle_gamma }
    symmetry { space_group_name_H_M }
    exptl_crystal_grow { pH temp method }
    polymer_entities {
      entity_poly { rcsb_entity_polymer_type rcsb_sample_sequence_length }
      rcsb_polymer_entity { pdbx_fragment pdbx_mutation }
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers { database_name database_accession }
      }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers {
          asym_id auth_asym_id entity_id
        }
      }
    }
    assemblies {
      rcsb_assembly_container_identifiers { assembly_id }
      rcsb_assembly_info {
        polymer_composition polymer_entity_instance_count
        polymer_entity_instance_count_protein nonpolymer_entity_instance_count
        total_assembly_buried_surface_area num_interfaces
      }
      pdbx_struct_assembly {
        details method_details oligomeric_count oligomeric_details rcsb_details
      }
      pdbx_struct_assembly_gen { asym_id_list oper_expression }
    }
  }
}
"""


def pdb_id_from_path(path: str | Path) -> str:
    """Extract a canonical four-character PDB ID from a coordinate filename."""
    stem = Path(path).stem.upper()
    match = re.match(r"^([0-9][A-Z0-9]{3})(?:[^A-Z0-9]|$)", stem)
    if match is None:
        raise ValueError(f"cannot infer PDB ID from coordinate path: {path}")
    return match.group(1)


def _finite_list(values: Iterable[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def parse_rcsb_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the subset of RCSB metadata used by environment audits."""
    chains: dict[str, list[dict[str, Any]]] = {}
    for entity in raw.get("polymer_entities") or []:
        entity_poly = entity.get("entity_poly") or {}
        if entity_poly.get("rcsb_entity_polymer_type") != "Protein":
            continue
        entity_details = entity.get("rcsb_polymer_entity") or {}
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        uniprot = sorted({
            str(item["database_accession"])
            for item in identifiers.get("reference_sequence_identifiers") or []
            if item.get("database_name") == "UniProt" and item.get("database_accession")
        })
        for instance in entity.get("polymer_entity_instances") or []:
            instance_ids = instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
            auth_chain = instance_ids.get("auth_asym_id")
            asym_id = instance_ids.get("asym_id")
            if not auth_chain or not asym_id:
                continue
            chains.setdefault(str(auth_chain), []).append({
                "asym_id": str(asym_id),
                "entity_id": str(instance_ids.get("entity_id", "")),
                "sample_sequence_length": entity_poly.get("rcsb_sample_sequence_length"),
                "fragment": entity_details.get("pdbx_fragment"),
                "mutation_annotation": entity_details.get("pdbx_mutation"),
                "uniprot_accessions": uniprot,
            })
    assemblies = []
    for assembly in raw.get("assemblies") or []:
        container = assembly.get("rcsb_assembly_container_identifiers") or {}
        info = assembly.get("rcsb_assembly_info") or {}
        details = assembly.get("pdbx_struct_assembly") or {}
        asym_ids = sorted({
            str(asym_id)
            for generator in assembly.get("pdbx_struct_assembly_gen") or []
            for asym_id in generator.get("asym_id_list") or []
        })
        assemblies.append({
            "assembly_id": str(container.get("assembly_id", "")),
            "asym_ids": asym_ids,
            "polymer_composition": info.get("polymer_composition"),
            "polymer_entity_instance_count": info.get("polymer_entity_instance_count"),
            "protein_instance_count": info.get("polymer_entity_instance_count_protein"),
            "nonpolymer_instance_count": info.get("nonpolymer_entity_instance_count"),
            "buried_surface_area": info.get("total_assembly_buried_surface_area"),
            "interface_count": info.get("num_interfaces"),
            "oligomeric_count": details.get("oligomeric_count"),
            "oligomeric_details": details.get("oligomeric_details"),
            "details": details.get("details"),
            "method_details": details.get("method_details"),
            "rcsb_details": details.get("rcsb_details"),
        })
    info = raw.get("rcsb_entry_info") or {}
    resolutions = _finite_list(info.get("resolution_combined") or [])
    cell = raw.get("cell") or {}
    crystal = raw.get("exptl_crystal_grow") or []
    return {
        "pdb_id": str(raw["rcsb_id"]).upper(),
        "title": (raw.get("struct") or {}).get("title"),
        "primary_citation": raw.get("rcsb_primary_citation"),
        "experimental_method": info.get("experimental_method"),
        "resolution_angstrom": min(resolutions) if resolutions else None,
        "chains": {key: value for key, value in sorted(chains.items())},
        "assemblies": assemblies,
        "space_group": (raw.get("symmetry") or {}).get("space_group_name_H_M"),
        "cell": {
            name: cell.get(name)
            for name in ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")
        } if cell else None,
        "crystal_growth": crystal,
    }


def target_profile(entry: dict[str, Any], auth_chain: str) -> dict[str, Any] | None:
    """Resolve an author chain to entity and biological-assembly metadata."""
    instances = entry.get("chains", {}).get(auth_chain) or []
    if not instances:
        return None
    asym_ids = {instance["asym_id"] for instance in instances}
    assemblies = [
        assembly
        for assembly in entry.get("assemblies", [])
        if asym_ids.intersection(assembly.get("asym_ids", []))
    ]
    signatures = sorted({
        (
            assembly.get("protein_instance_count"),
            assembly.get("polymer_composition"),
            assembly.get("oligomeric_count"),
        )
        for assembly in assemblies
        if assembly.get("protein_instance_count") is not None
    }, key=str)
    return {
        "auth_chain": auth_chain,
        "asym_ids": sorted(asym_ids),
        "entity_ids": sorted({instance["entity_id"] for instance in instances}),
        "uniprot_accessions": sorted({
            accession
            for instance in instances
            for accession in instance.get("uniprot_accessions", [])
        }),
        "sample_sequence_lengths": sorted({
            int(instance["sample_sequence_length"])
            for instance in instances
            if instance.get("sample_sequence_length") is not None
        }),
        "fragments": sorted({
            str(instance["fragment"])
            for instance in instances
            if instance.get("fragment")
        }),
        "mutation_annotations": sorted({
            str(instance["mutation_annotation"])
            for instance in instances
            if instance.get("mutation_annotation")
        }),
        "assembly_ids": sorted(assembly["assembly_id"] for assembly in assemblies),
        "assembly_signatures": [list(signature) for signature in signatures],
    }


def cell_compatible(
    parent: dict[str, Any],
    target: dict[str, Any],
    *,
    relative_length_tolerance: float,
    angle_tolerance: float,
) -> tuple[bool, dict[str, float] | None]:
    """Compare crystallographic unit cells using explicit tolerances."""
    parent_cell, target_cell = parent.get("cell"), target.get("cell")
    if not parent_cell or not target_cell:
        return False, None
    lengths = ("length_a", "length_b", "length_c")
    angles = ("angle_alpha", "angle_beta", "angle_gamma")
    values = [parent_cell.get(name) for name in lengths + angles]
    values += [target_cell.get(name) for name in lengths + angles]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        return False, None
    relative = max(
        abs(float(parent_cell[name]) - float(target_cell[name])) / max(float(parent_cell[name]), 1e-12)
        for name in lengths
    )
    angle = max(abs(float(parent_cell[name]) - float(target_cell[name])) for name in angles)
    return relative <= relative_length_tolerance and angle <= angle_tolerance, {
        "max_relative_length_difference": relative,
        "max_angle_difference_degree": angle,
    }


def load_metadata_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text())
    if payload.get("format") not in LEGACY_CACHE_FORMATS | {CACHE_FORMAT}:
        raise ValueError(f"metadata cache must use {CACHE_FORMAT}")
    return {str(key).upper(): value for key, value in payload.get("entries", {}).items()}


def _write_cache(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "format": CACHE_FORMAT,
        "entries": dict(sorted(entries.items())),
    }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch_rcsb_environment_metadata(
    entry_ids: Iterable[str],
    *,
    batch_size: int = 50,
    timeout: float = 30.0,
    retries: int = 2,
    existing: dict[str, dict[str, Any]] | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch missing entry metadata in bounded batches and atomically cache it."""
    if batch_size <= 0 or timeout <= 0 or retries < 0:
        raise ValueError("batch_size and timeout must be positive; retries must be non-negative")
    output = {str(key).upper(): value for key, value in (existing or {}).items()}
    identifiers = sorted({
        value.strip().upper()
        for value in entry_ids
        if value.strip()
        and (
            value.strip().upper() not in output
            or "title" not in output[value.strip().upper()]
        )
    })
    destination = Path(cache_path) if cache_path is not None else None
    for start in range(0, len(identifiers), batch_size):
        batch = identifiers[start : start + batch_size]
        request = Request(
            GRAPHQL_ENDPOINT,
            data=json.dumps({"query": ENTRY_QUERY, "variables": {"ids": batch}}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "ospedit/0.1"},
        )
        payload = None
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:  # noqa: S310
                    payload = json.load(response)
                last_error = None
                break
            except (HTTPError, URLError, OSError) as error:
                last_error = error
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        if payload is None:
            assert last_error is not None
            raise last_error
        if payload.get("errors"):
            raise ValueError(f"RCSB GraphQL error: {payload['errors']}")
        for raw in (payload.get("data") or {}).get("entries") or []:
            parsed = parse_rcsb_entry(raw)
            output[parsed["pdb_id"]] = parsed
        if destination is not None:
            _write_cache(destination, output)
    return output
