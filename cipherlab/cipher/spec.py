from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


Architecture = Literal["SPN", "FEISTEL", "ARX"]


class CipherSpec(BaseModel):
    """A *research* block-cipher spec for the builder/exporter.

    This is NOT a guarantee of security. It is a structured description
    of a cipher construction so we can:
    - generate Python code
    - run local metrics
    - iterate improvements reproducibly
    """

    name: str = Field(..., min_length=3, max_length=80)
    architecture: Architecture
    block_size_bits: int = Field(..., description="64 or 128 recommended", ge=32, le=256)
    key_size_bits: int = Field(..., ge=32, le=512)
    rounds: int = Field(..., ge=1, le=64)

    # Map stage -> component_id (registered in ComponentRegistry)
    components: Dict[str, str] = Field(default_factory=dict)

    version: str = Field(default="0.1")
    notes: str = Field(default="")
    seed: int = Field(default=1337, description="Used for deterministic key schedule and tests")

    @field_validator("architecture")
    @classmethod
    def _upper_arch(cls, v: str) -> str:
        return v.upper()

    @field_validator("block_size_bits")
    @classmethod
    def _block_size(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("block_size_bits must be a multiple of 8")
        return v

    @field_validator("key_size_bits")
    @classmethod
    def _key_size(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("key_size_bits must be a multiple of 8")
        return v


class ImprovementPatch(BaseModel):
    """A small patch to apply to an existing CipherSpec."""

    summary: str = Field(..., min_length=5, max_length=240)
    rationale: List[str] = Field(default_factory=list, description="Bullet reasons / design principles")

    new_rounds: Optional[int] = Field(default=None, ge=1, le=64)
    replace_components: Optional[Dict[str, str]] = Field(
        default=None,
        description="stage -> new component_id",
    )
    add_notes: Optional[str] = Field(default=None, max_length=2000)
