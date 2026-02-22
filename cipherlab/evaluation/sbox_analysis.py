"""S-box differential and linear analysis wrappers.

Wraps the existing sbox_ddt_max and sbox_lat_max_abs from
cipherlab.cipher.cryptanalysis with structured result output
and bijectivity checking.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from cipherlab.cipher.cryptanalysis import sbox_ddt_max, sbox_lat_max_abs
from cipherlab.cipher.registry import ComponentRegistry


@dataclass
class SBoxAnalysisResult:
    """Structured result of S-box differential/linear analysis."""
    component_id: str
    sbox_size: int              # 16 (4-bit) or 256 (8-bit)
    ddt_max: int                # Max DDT entry (ideal: 2 for 4-bit, 4 for 8-bit)
    lat_max_abs: int            # Max LAT absolute bias (lower = better)
    is_bijective: bool          # Forward then inverse = identity
    differential_uniformity: str  # "good" / "fair" / "poor"
    linearity: str              # "good" / "fair" / "poor"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        bij = "bijective" if self.is_bijective else "NOT bijective"
        return (
            f"{self.component_id} ({self.sbox_size}-entry): "
            f"DDT_max={self.ddt_max} ({self.differential_uniformity}), "
            f"LAT_max={self.lat_max_abs} ({self.linearity}), {bij}"
        )


def _extract_sbox_table(comp, registry: ComponentRegistry) -> List[int]:
    """Extract the S-box lookup table by enumerating all input values.

    For 4-bit S-boxes (PRESENT, GIFT): process nibble-by-nibble.
    For 8-bit S-boxes (AES, etc.): process byte-by-byte.
    """
    fwd = comp.forward

    # Detect 4-bit vs 8-bit by component_id naming convention
    is_4bit = any(tag in comp.component_id for tag in ("present", "gift"))

    if is_4bit:
        # 4-bit S-box: feed single bytes with value 0x00..0x0F
        table = []
        for nibble_val in range(16):
            result = fwd(bytes([nibble_val]))
            table.append(result[0] & 0x0F)  # Low nibble output
        return table
    else:
        # 8-bit S-box: feed single bytes with value 0x00..0xFF
        table = []
        for byte_val in range(256):
            result = fwd(bytes([byte_val]))
            table.append(result[0])
        return table


def _check_bijectivity(comp) -> bool:
    """Check if forward(inverse(x)) == x for all input values."""
    if comp.inverse is None:
        return False

    fwd = comp.forward
    inv = comp.inverse

    is_4bit = any(tag in comp.component_id for tag in ("present", "gift"))
    max_val = 16 if is_4bit else 256

    for val in range(max_val):
        original = bytes([val])
        try:
            encrypted = fwd(original)
            decrypted = inv(encrypted)
            if is_4bit:
                if (decrypted[0] & 0x0F) != (original[0] & 0x0F):
                    return False
            else:
                if decrypted[0] != original[0]:
                    return False
        except Exception:
            return False
    return True


def _rate_differential_uniformity(ddt_max: int, sbox_size: int) -> str:
    """Rate DDT max value quality."""
    if sbox_size == 16:  # 4-bit S-box
        if ddt_max <= 4:
            return "good"
        elif ddt_max <= 6:
            return "fair"
        else:
            return "poor"
    else:  # 8-bit S-box (256 entries)
        if ddt_max <= 4:
            return "good"
        elif ddt_max <= 8:
            return "fair"
        else:
            return "poor"


def _rate_linearity(lat_max: int, sbox_size: int) -> str:
    """Rate LAT max absolute bias quality."""
    if sbox_size == 16:  # 4-bit
        if lat_max <= 4:
            return "good"
        elif lat_max <= 6:
            return "fair"
        else:
            return "poor"
    else:  # 8-bit
        if lat_max <= 16:
            return "good"
        elif lat_max <= 32:
            return "fair"
        else:
            return "poor"


def analyze_sbox(
    component_id: str,
    registry: Optional[ComponentRegistry] = None,
) -> SBoxAnalysisResult:
    """Analyze a single S-box component for differential/linear properties.

    Args:
        component_id: Registry ID of the S-box component.
        registry: Optional component registry; uses default if not provided.

    Returns:
        SBoxAnalysisResult with DDT max, LAT max, bijectivity, and ratings.
    """
    reg = registry or ComponentRegistry()

    if not reg.exists(component_id):
        raise ValueError(f"Unknown component: {component_id}")

    comp = reg.get(component_id)
    table = _extract_sbox_table(comp, reg)
    sbox_size = len(table)

    ddt = sbox_ddt_max(table)
    lat = sbox_lat_max_abs(table)
    bijective = _check_bijectivity(comp)

    return SBoxAnalysisResult(
        component_id=component_id,
        sbox_size=sbox_size,
        ddt_max=ddt,
        lat_max_abs=lat,
        is_bijective=bijective,
        differential_uniformity=_rate_differential_uniformity(ddt, sbox_size),
        linearity=_rate_linearity(lat, sbox_size),
    )


def analyze_all_sboxes(
    registry: Optional[ComponentRegistry] = None,
) -> List[SBoxAnalysisResult]:
    """Analyze all S-box components in the registry.

    Skips Feistel F-function components (tea_f, xtea_f, simon_f, hight_f)
    since they are not traditional substitution boxes.

    Returns:
        List of SBoxAnalysisResult for each analyzable S-box.
    """
    reg = registry or ComponentRegistry()
    results: List[SBoxAnalysisResult] = []

    # Get all SBOX kind components
    sbox_comps = reg.list_by_kind("SBOX")

    # Skip Feistel F-functions (not traditional S-boxes with lookup tables)
    skip_ids = {"sbox.tea_f", "sbox.xtea_f", "sbox.simon_f", "sbox.hight_f"}

    for comp in sbox_comps:
        if comp.component_id in skip_ids:
            continue
        # Also skip identity S-box (trivial)
        if comp.component_id == "sbox.identity":
            continue
        try:
            result = analyze_sbox(comp.component_id, reg)
            results.append(result)
        except Exception:
            continue

    return sorted(results, key=lambda r: r.component_id)
