"""Built-in cipher components for the cipherlab package.

This module re-exports components from the main Components.py file
and provides the Component dataclass and builtins() function.

Research / education only. Do NOT use in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for importing Components
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import everything from the main Components module
from Components import (
    Component,
    ComponentRegistry,
    CipherConfiguration,
    builtin_components,
    xor_bytes,
    # Key schedules
    ks_sha256_kdf,
    ks_des_style,
    ks_blowfish_style,
    # S-boxes (original)
    sbox_aes, sboxinv_aes,
    sbox_identity,
    sbox_des, sboxinv_des,
    sbox_blowfish, sboxinv_blowfish,
    sbox_serpent, sboxinv_serpent,
    # S-boxes (LWC Phase 2)
    sbox_present, sboxinv_present,
    sbox_gift, sboxinv_gift,
    sbox_tea_f,
    sbox_xtea_f,
    sbox_simon_f,
    sbox_hight_f,
    # Permutations (original)
    perm_aes_shiftrows, perm_aes_inv_shiftrows,
    perm_identity,
    perm_des_ip, perm_des_ip_inv,
    perm_serpent, perm_serpent_inv,
    # Permutations (LWC Phase 2)
    perm_present, perm_present_inv,
    perm_gift, perm_gift_inv,
    # Linear layers
    linear_aes_mixcolumns, linear_aes_inv_mixcolumns,
    linear_identity,
    linear_twofish_mds, linear_twofish_mds_inv,
    # ARX operations
    arx_add_mod32, arx_sub_mod32,
    arx_rotate_left_5, arx_rotate_right_5,
    arx_rotate_left_3, arx_rotate_right_3,
    arx_mul_mod16, arx_mul_mod16_inv,
    # ARX factory functions
    make_arx_add, make_arx_rotate,
)


def builtins():
    """Return all built-in cipher components for the registry."""
    return builtin_components()
