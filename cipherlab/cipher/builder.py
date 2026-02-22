from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .components_builtin import xor_bytes
from .registry import ComponentRegistry
from .spec import CipherSpec


class BlockCipher:
    def encrypt_block(self, plaintext_block: bytes, key: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def decrypt_block(self, ciphertext_block: bytes, key: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError


@dataclass
class SPNCipher(BlockCipher):
    spec: CipherSpec
    key_schedule: Callable
    sbox_fwd: Callable
    sbox_inv: Callable
    perm_fwd: Callable
    perm_inv: Callable
    lin_fwd: Callable
    lin_inv: Callable

    def encrypt_block(self, plaintext_block: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(plaintext_block) != bs:
            raise ValueError(f"Plaintext block must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        round_keys: List[bytes] = self.key_schedule(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = plaintext_block

        for r in range(self.spec.rounds):
            state = xor_bytes(state, round_keys[r])
            state = self.sbox_fwd(state)
            state = self.perm_fwd(state)
            # Standard SPN often omits linear layer in last round; we do the same.
            if r != self.spec.rounds - 1:
                state = self.lin_fwd(state)

        state = xor_bytes(state, round_keys[self.spec.rounds])
        return state

    def decrypt_block(self, ciphertext_block: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(ciphertext_block) != bs:
            raise ValueError(f"Ciphertext block must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        round_keys: List[bytes] = self.key_schedule(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = xor_bytes(ciphertext_block, round_keys[self.spec.rounds])

        for r in reversed(range(self.spec.rounds)):
            if r != self.spec.rounds - 1:
                state = self.lin_inv(state)
            state = self.perm_inv(state)
            state = self.sbox_inv(state)
            state = xor_bytes(state, round_keys[r])

        return state


@dataclass
class FeistelCipher(BlockCipher):
    spec: CipherSpec
    key_schedule: Callable
    f_sbox: Callable
    f_perm: Callable

    def encrypt_block(self, plaintext_block: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(plaintext_block) != bs:
            raise ValueError(f"Plaintext block must be {bs} bytes")
        if bs % 2 != 0:
            raise ValueError("Feistel requires even byte block size")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        half = bs // 2
        round_keys: List[bytes] = self.key_schedule(key, rounds=self.spec.rounds, out_len=half, seed=self.spec.seed)
        L, R = plaintext_block[:half], plaintext_block[half:]

        for r in range(self.spec.rounds):
            F = self.f_perm(self.f_sbox(xor_bytes(R, round_keys[r])))
            L, R = R, xor_bytes(L, F)

        return L + R

    def decrypt_block(self, ciphertext_block: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(ciphertext_block) != bs:
            raise ValueError(f"Ciphertext block must be {bs} bytes")
        if bs % 2 != 0:
            raise ValueError("Feistel requires even byte block size")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        half = bs // 2
        round_keys: List[bytes] = self.key_schedule(key, rounds=self.spec.rounds, out_len=half, seed=self.spec.seed)
        L, R = ciphertext_block[:half], ciphertext_block[half:]

        for r in reversed(range(self.spec.rounds)):
            # Reverse of: L,R = R, L XOR F(R, k)
            # so: prev_R = L
            #     prev_L = R XOR F(prev_R, k)
            prev_R = L
            F = self.f_perm(self.f_sbox(xor_bytes(prev_R, round_keys[r])))
            prev_L = xor_bytes(R, F)
            L, R = prev_L, prev_R

        return L + R


@dataclass
class ARXCipher(BlockCipher):
    """Add-Rotate-XOR cipher implementation.
    
    ARX structure used by ciphers like RC5, RC6, IDEA, ChaCha:
    1. XOR with round key
    2. Apply ARX operations (modular addition, rotation)
    3. Repeat for all rounds
    """
    spec: CipherSpec
    key_schedule: Callable
    arx_add_fwd: Callable
    arx_add_inv: Callable
    arx_rot_fwd: Callable
    arx_rot_inv: Callable

    def encrypt_block(self, plaintext_block: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(plaintext_block) != bs:
            raise ValueError(f"Plaintext block must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        round_keys: List[bytes] = self.key_schedule(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = plaintext_block

        for r in range(self.spec.rounds):
            state = xor_bytes(state, round_keys[r])
            state = self.arx_add_fwd(state)
            state = self.arx_rot_fwd(state)

        state = xor_bytes(state, round_keys[self.spec.rounds])
        return state

    def decrypt_block(self, ciphertext_block: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(ciphertext_block) != bs:
            raise ValueError(f"Ciphertext block must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        round_keys: List[bytes] = self.key_schedule(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = xor_bytes(ciphertext_block, round_keys[self.spec.rounds])

        for r in reversed(range(self.spec.rounds)):
            state = self.arx_rot_inv(state)
            state = self.arx_add_inv(state)
            state = xor_bytes(state, round_keys[r])

        return state


def build_cipher(spec: CipherSpec, registry: Optional[ComponentRegistry] = None) -> BlockCipher:
    reg = registry or ComponentRegistry()

    ks_id = spec.components.get("key_schedule", "ks.sha256_kdf")
    ks = reg.get(ks_id).forward

    if spec.architecture == "SPN":
        sbox_id = spec.components.get("sbox", "sbox.aes")
        perm_id = spec.components.get("perm", "perm.aes_shiftrows")
        lin_id = spec.components.get("linear", "linear.aes_mixcolumns")

        sbox = reg.get(sbox_id)
        perm = reg.get(perm_id)
        lin = reg.get(lin_id)
        if not (sbox.inverse and perm.inverse and lin.inverse):
            raise ValueError("SPN requires invertible components (inverse must be provided)")

        return SPNCipher(
            spec=spec,
            key_schedule=ks,
            sbox_fwd=sbox.forward,
            sbox_inv=sbox.inverse,
            perm_fwd=perm.forward,
            perm_inv=perm.inverse,
            lin_fwd=lin.forward,
            lin_inv=lin.inverse,
        )

    if spec.architecture == "FEISTEL":
        f_sbox_id = spec.components.get("f_sbox", "sbox.aes")
        f_perm_id = spec.components.get("f_perm", "perm.identity")
        f_sbox = reg.get(f_sbox_id).forward
        f_perm = reg.get(f_perm_id).forward
        return FeistelCipher(spec=spec, key_schedule=ks, f_sbox=f_sbox, f_perm=f_perm)

    if spec.architecture == "ARX":
        # ARX ciphers like RC5, RC6, IDEA
        arx_add_id = spec.components.get("arx_add", "arx.add_mod32")
        arx_rot_id = spec.components.get("arx_rotate", "arx.rotate_left_5")

        arx_add = reg.get(arx_add_id)
        arx_rot = reg.get(arx_rot_id)
        if not (arx_add.inverse and arx_rot.inverse):
            raise ValueError("ARX requires invertible components (inverse must be provided)")

        return ARXCipher(
            spec=spec,
            key_schedule=ks,
            arx_add_fwd=arx_add.forward,
            arx_add_inv=arx_add.inverse,
            arx_rot_fwd=arx_rot.forward,
            arx_rot_inv=arx_rot.inverse,
        )

    raise ValueError(f"Unknown architecture: {spec.architecture}")

