from cipherlab.cipher.spec import CipherSpec
from cipherlab.cipher.builder import build_cipher


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
