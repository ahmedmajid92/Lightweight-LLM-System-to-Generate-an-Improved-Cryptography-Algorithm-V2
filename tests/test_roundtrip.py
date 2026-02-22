import sys
from pathlib import Path

import pytest

# Ensure project root is on path for AlgorithmsBlock imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cipherlab.cipher.spec import CipherSpec
from cipherlab.cipher.builder import build_cipher
from AlgorithmsBlock import (
    get_template,
    list_algorithms,
    build_cipher as ab_build_cipher,
    ComponentRegistry,
)


# ---------------------------------------------------------------------------
# Original hand-crafted tests (preserved from Phase 1)
# ---------------------------------------------------------------------------

def test_spn_roundtrip():
    spec = CipherSpec(
        name="TestSPN",
        architecture="SPN",
        block_size_bits=128,
        key_size_bits=128,
        rounds=10,
        components={
            "sbox": "sbox.aes",
            "perm": "perm.aes_shiftrows",
            "linear": "linear.aes_mixcolumns",
            "key_schedule": "ks.sha256_kdf",
        },
        seed=1337,
    )
    cipher = build_cipher(spec)
    key = b"K" * 16
    pt = bytes(range(16))
    ct = cipher.encrypt_block(pt, key)
    rt = cipher.decrypt_block(ct, key)
    assert rt == pt


def test_feistel_roundtrip():
    spec = CipherSpec(
        name="TestFeistel",
        architecture="FEISTEL",
        block_size_bits=64,
        key_size_bits=128,
        rounds=16,
        components={
            "f_sbox": "sbox.aes",
            "f_perm": "perm.identity",
            "key_schedule": "ks.sha256_kdf",
        },
        seed=2026,
    )
    cipher = build_cipher(spec)
    key = b"K" * 16
    pt = bytes(range(8))
    ct = cipher.encrypt_block(pt, key)
    rt = cipher.decrypt_block(ct, key)
    assert rt == pt


def test_arx_roundtrip():
    """ARX roundtrip test (previously missing)."""
    spec = CipherSpec(
        name="TestARX",
        architecture="ARX",
        block_size_bits=64,
        key_size_bits=128,
        rounds=12,
        components={
            "arx_add": "arx.add_mod32",
            "arx_rotate": "arx.rotate_left_5",
            "key_schedule": "ks.sha256_kdf",
        },
        seed=42,
    )
    cipher = build_cipher(spec)
    key = b"K" * 16
    pt = bytes(range(8))
    ct = cipher.encrypt_block(pt, key)
    rt = cipher.decrypt_block(ct, key)
    assert rt == pt


# ---------------------------------------------------------------------------
# Parametrized test: all 12 algorithms via AlgorithmsBlock templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo_name", list_algorithms())
def test_all_algorithms_roundtrip(algo_name):
    """Roundtrip P = D(E(P, K), K) for each algorithm in ALGORITHM_LIBRARY."""
    import random

    spec = get_template(algo_name, seed=1337)
    reg = ComponentRegistry()
    cipher = ab_build_cipher(spec, reg)

    block_bytes = spec.block_size_bits // 8
    key_bytes = spec.key_size_bits // 8

    rng = random.Random(1337)

    # Test with 50 random vectors per algorithm
    for _ in range(50):
        pt = bytes(rng.randrange(0, 256) for _ in range(block_bytes))
        key = bytes(rng.randrange(0, 256) for _ in range(key_bytes))
        ct = cipher.encrypt_block(pt, key)
        rt = cipher.decrypt_block(ct, key)
        assert rt == pt, (
            f"{algo_name}: roundtrip failed. "
            f"pt={pt.hex()}, key={key.hex()}, ct={ct.hex()}, rt={rt.hex()}"
        )
