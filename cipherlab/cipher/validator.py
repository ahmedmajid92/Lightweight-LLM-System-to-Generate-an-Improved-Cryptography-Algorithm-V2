from __future__ import annotations

from typing import List, Tuple

from .registry import ComponentRegistry
from .spec import CipherSpec


def validate_spec(spec: CipherSpec, registry: ComponentRegistry | None = None) -> Tuple[bool, List[str]]:
    reg = registry or ComponentRegistry()
    errs: List[str] = []

    # Generic checks
    if spec.block_size_bits % 8 != 0:
        errs.append("block_size_bits must be multiple of 8")
    if spec.key_size_bits % 8 != 0:
        errs.append("key_size_bits must be multiple of 8")

    # Required key schedule
    ks_id = spec.components.get("key_schedule", "ks.sha256_kdf")
    if not reg.exists(ks_id):
        errs.append(f"Unknown key_schedule component: {ks_id}")

    if spec.architecture == "SPN":
        if spec.block_size_bits != 128:
            errs.append("SPN template currently supports block_size_bits=128 only")
        for k in ["sbox", "perm", "linear"]:
            if k not in spec.components:
                errs.append(f"Missing SPN component: {k}")
            else:
                cid = spec.components[k]
                if not reg.exists(cid):
                    errs.append(f"Unknown component {k}: {cid}")

    elif spec.architecture == "FEISTEL":
        bs = spec.block_size_bits // 8
        if bs % 2 != 0:
            errs.append("Feistel requires even-byte block size")
        for k in ["f_sbox", "f_perm"]:
            if k not in spec.components:
                errs.append(f"Missing FEISTEL component: {k}")
            else:
                cid = spec.components[k]
                if not reg.exists(cid):
                    errs.append(f"Unknown component {k}: {cid}")
    else:
        errs.append(f"Unsupported architecture: {spec.architecture}")

    return (len(errs) == 0), errs
