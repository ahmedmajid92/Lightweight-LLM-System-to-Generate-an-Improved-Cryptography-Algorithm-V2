from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from cipherlab.config import load_settings


def make_example(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    # Minimal synthetic dataset generator (v2)
    system_general = "You are a careful cryptography tutor and block-cipher engineer. Stay accurate and avoid unsupported claims."
    system_spec = "You generate CipherSpec JSON for Crypto Cipher Lab v2. Output ONLY JSON for CipherSpec."
    system_patch = "You generate ImprovementPatch JSON for Crypto Cipher Lab v2. Output ONLY JSON for ImprovementPatch."
    system_code = "You write Python code (no markdown) using the Crypto Cipher Lab v2 project APIs."

    algos = [
        {
            "name": "AES",
            "architecture": "SPN",
            "block_bits": 128,
            "key_bits": "128/192/256",
            "rounds": "10/12/14",
            "components": [
                "8-bit S-box (SubBytes)",
                "ShiftRows permutation",
                "MixColumns linear diffusion",
                "AddRoundKey (XOR)",
                "Key schedule / round key expansion",
            ],
            "notes": [
                "Widely standardized by NIST (FIPS 197).",
                "Reduced-round variants have published attacks; full-round AES remains widely trusted.",
            ],
        },
        # (Other algorithms omitted here for brevity — edit if you want more)
    ]

    examples: List[dict] = []

    # A few KB-style summaries
    for a in algos:
        user = f"Give me a high-level summary of {a['name']} for a block-cipher knowledge base."
        assistant = f"""# {a['name']}\n\n- Architecture: {a['architecture']}\n- Block: {a['block_bits']} bits\n- Keys: {a['key_bits']}\n- Rounds: {a['rounds']}\n\nComponents:\n""" + "\n".join(["- " + c for c in a["components"]])
        examples.append(make_example(system_general, user, assistant))

    # CipherSpec examples
    for i in range(40):
        if i % 2 == 0:
            spec = {
                "name": f"SPN_Lab_{i}",
                "architecture": "SPN",
                "block_size_bits": 128,
                "key_size_bits": 128,
                "rounds": 10,
                "components": {"sbox": "sbox.aes", "perm": "perm.aes_shiftrows", "linear": "linear.aes_mixcolumns", "key_schedule": "ks.sha256_kdf"},
                "version": "0.1",
                "notes": "",
                "seed": 1337,
            }
        else:
            spec = {
                "name": f"FEISTEL_Lab_{i}",
                "architecture": "FEISTEL",
                "block_size_bits": 64,
                "key_size_bits": 128,
                "rounds": 16,
                "components": {"f_sbox": "sbox.aes", "f_perm": "perm.identity", "key_schedule": "ks.sha256_kdf"},
                "version": "0.1",
                "notes": "",
                "seed": 2026,
            }
        user = "Create a CipherSpec.\n" + json.dumps(spec, indent=2)
        examples.append(make_example(system_spec, user, json.dumps(spec, indent=2)))

    # ImprovementPatch examples
    for i in range(30):
        patch = {
            "summary": "Increase rounds and add nonlinearity to improve avalanche.",
            "rationale": ["More rounds tends to improve diffusion in toy ciphers.", "Adding a nonlinear S-box improves confusion."],
            "new_rounds": 16,
            "replace_components": {"sbox": "sbox.aes"},
            "add_notes": "Hypothesis only; not a proof of security.",
        }
        user = "Propose an ImprovementPatch for a weak cipher (details omitted)."
        examples.append(make_example(system_patch, user, json.dumps(patch, indent=2)))

    # Code examples using exporter
    for i in range(20):
        name = f"AutoSPN_{i}"
        spec = {
            "name": name,
            "architecture": "SPN",
            "block_size_bits": 128,
            "key_size_bits": 128,
            "rounds": 10,
            "components": {"sbox": "sbox.aes", "perm": "perm.aes_shiftrows", "linear": "linear.aes_mixcolumns", "key_schedule": "ks.sha256_kdf"},
            "version": "0.1",
            "notes": "",
            "seed": 1337,
        }
        user = "Write code to export a cipher module and run self_test. Spec:\n" + json.dumps(spec, indent=2)
        assistant = f'''from cipherlab.cipher.spec import CipherSpec\nfrom cipherlab.cipher.exporter import export_cipher_module\n\nspec = CipherSpec(**{json.dumps(spec)})\ncode = export_cipher_module(spec)\n\nout_path = "{name}_cipher.py"\nwith open(out_path, "w", encoding="utf-8") as f:\n    f.write(code)\n\nns = {{}}\nexec(compile(code, out_path, "exec"), ns, ns)\nns["self_test"]()\n'''
        examples.append(make_example(system_code, user, assistant))

    rng = random.Random(42)
    rng.shuffle(examples)
    split = int(len(examples) * 0.9)
    train, valid = examples[:split], examples[split:]

    settings = load_settings()
    out_dir = Path(settings.project_root) / "data" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)

    print(f"Wrote {len(train)} train and {len(valid)} valid examples to {out_dir}")


if __name__ == "__main__":
    main()
