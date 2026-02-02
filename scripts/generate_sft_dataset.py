"""Generate 500 balanced SFT training examples for the block cipher LLM.

This script creates a comprehensive dataset covering:
1. CipherSpec generation (all 12 algorithms)
2. ImprovementPatch suggestions
3. Python code generation with comments
4. KB summaries and comparisons
5. Q&A about block cipher concepts

Run: python scripts/generate_sft_dataset.py
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sft"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALID_FILE = OUTPUT_DIR / "valid.jsonl"
TRAIN_SIZE = 450
VALID_SIZE = 50

# All 12 algorithms with their specifications
ALGORITHMS = {
    "AES": {"arch": "SPN", "block": 128, "key": [128, 256], "rounds": [10, 12, 14]},
    "DES": {"arch": "FEISTEL", "block": 64, "key": [128], "rounds": [16]},
    "3DES": {"arch": "FEISTEL", "block": 64, "key": [128], "rounds": [48]},
    "Blowfish": {"arch": "FEISTEL", "block": 64, "key": [128, 256], "rounds": [16]},
    "Twofish": {"arch": "FEISTEL", "block": 128, "key": [128, 256], "rounds": [16]},
    "Serpent": {"arch": "SPN", "block": 128, "key": [128, 256], "rounds": [32]},
    "Camellia": {"arch": "FEISTEL", "block": 128, "key": [128, 256], "rounds": [18, 24]},
    "CAST-128": {"arch": "FEISTEL", "block": 64, "key": [128], "rounds": [12, 16]},
    "IDEA": {"arch": "ARX", "block": 64, "key": [128], "rounds": [8]},
    "SEED": {"arch": "FEISTEL", "block": 128, "key": [128], "rounds": [16]},
    "RC5": {"arch": "ARX", "block": 64, "key": [128], "rounds": [12, 16, 20]},
    "RC6": {"arch": "ARX", "block": 128, "key": [128, 256], "rounds": [20]},
}

# Components by architecture
SPN_COMPONENTS = {
    "sbox": ["sbox.aes", "sbox.serpent"],
    "perm": ["perm.aes_shiftrows", "perm.serpent", "perm.identity"],
    "linear": ["linear.aes_mixcolumns", "linear.identity"],
    "key_schedule": ["ks.sha256_kdf"],
}

FEISTEL_COMPONENTS = {
    "f_sbox": ["sbox.aes", "sbox.des", "sbox.blowfish", "sbox.identity"],
    "f_perm": ["perm.identity", "perm.des_ip"],
    "key_schedule": ["ks.sha256_kdf", "ks.des_style", "ks.blowfish_style"],
}

ARX_COMPONENTS = {
    "arx_add": ["arx.add_mod32", "arx.mul_mod16"],
    "arx_rotate": ["arx.rotate_left_3", "arx.rotate_left_5"],
    "key_schedule": ["ks.sha256_kdf"],
}

# System prompts
SYSTEM_PROMPTS = {
    "spec": "You generate CipherSpec JSON for Crypto Cipher Lab v2. Output ONLY valid JSON for CipherSpec. Include helpful comments in the notes field.",
    "patch": "You generate ImprovementPatch JSON for Crypto Cipher Lab v2. Output ONLY valid JSON for ImprovementPatch. Provide clear rationale for improvements.",
    "code": "You write Python code for Crypto Cipher Lab v2. Include detailed comments explaining each step. No markdown, just raw Python code.",
    "tutor": "You are a careful cryptography tutor and block-cipher engineer. Stay accurate and avoid unsupported security claims. For research/education only.",
    "qa": "You answer questions about block ciphers clearly and accurately. Reference specific algorithms and components when relevant. For research/education only.",
}


def make_message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


def make_example(system: str, user: str, assistant: str) -> Dict[str, List[Dict[str, str]]]:
    return {"messages": [
        make_message("system", system),
        make_message("user", user),
        make_message("assistant", assistant),
    ]}


def gen_cipherspec_example(algo: str, seed: int) -> Dict:
    """Generate a CipherSpec creation example."""
    info = ALGORITHMS[algo]
    arch = info["arch"]
    block = info["block"]
    key = random.choice(info["key"])
    rounds = random.choice(info["rounds"])
    
    # Select components based on architecture
    if arch == "SPN":
        components = {
            "sbox": random.choice(SPN_COMPONENTS["sbox"]),
            "perm": random.choice(SPN_COMPONENTS["perm"]),
            "linear": random.choice(SPN_COMPONENTS["linear"]),
            "key_schedule": random.choice(SPN_COMPONENTS["key_schedule"]),
        }
    elif arch == "FEISTEL":
        components = {
            "f_sbox": random.choice(FEISTEL_COMPONENTS["f_sbox"]),
            "f_perm": random.choice(FEISTEL_COMPONENTS["f_perm"]),
            "key_schedule": random.choice(FEISTEL_COMPONENTS["key_schedule"]),
        }
    else:  # ARX
        components = {
            "arx_add": random.choice(ARX_COMPONENTS["arx_add"]),
            "arx_rotate": random.choice(ARX_COMPONENTS["arx_rotate"]),
            "key_schedule": random.choice(ARX_COMPONENTS["key_schedule"]),
        }
    
    name = f"{algo}_Spec_{seed}"
    
    user_prompt = f"""Create a CipherSpec for a {arch} cipher based on {algo}.
Requirements:
- Name: {name}
- Block size: {block} bits
- Key size: {key} bits
- Rounds: {rounds}
- Use seed={seed} for reproducibility

Choose appropriate components for the {arch} architecture."""

    spec_json = {
        "name": name,
        "architecture": arch,
        "block_size_bits": block,
        "key_size_bits": key,
        "rounds": rounds,
        "components": components,
        "version": "0.2",
        "notes": f"{algo}-style {arch} cipher for research. Not for production use.",
        "seed": seed
    }
    
    return make_example(
        SYSTEM_PROMPTS["spec"],
        user_prompt,
        json.dumps(spec_json, indent=2)
    )


def gen_improvement_example(algo: str, seed: int) -> Dict:
    """Generate an ImprovementPatch example."""
    info = ALGORITHMS[algo]
    arch = info["arch"]
    
    # Create a "weak" spec with identity components
    if arch == "SPN":
        weak_components = {"sbox": "sbox.identity", "perm": "perm.identity", "linear": "linear.identity", "key_schedule": "ks.sha256_kdf"}
        improved_components = {"sbox": "sbox.aes", "perm": "perm.aes_shiftrows", "linear": "linear.aes_mixcolumns"}
    elif arch == "FEISTEL":
        weak_components = {"f_sbox": "sbox.identity", "f_perm": "perm.identity", "key_schedule": "ks.sha256_kdf"}
        improved_components = {"f_sbox": "sbox.aes"}
    else:  # ARX
        weak_components = {"arx_add": "arx.add_mod32", "arx_rotate": "arx.rotate_left_3", "key_schedule": "ks.sha256_kdf"}
        improved_components = {"arx_rotate": "arx.rotate_left_5"}
    
    low_avalanche = round(random.uniform(0.2, 0.35), 2)
    
    spec = {
        "name": f"{algo}_Weak_{seed}",
        "architecture": arch,
        "block_size_bits": info["block"],
        "key_size_bits": random.choice(info["key"]),
        "rounds": 8,
        "components": weak_components,
        "version": "0.2",
        "notes": "",
        "seed": seed
    }
    
    metrics = {
        "plaintext_avalanche": {"mean": low_avalanche},
        "key_avalanche": {"mean": low_avalanche + 0.02}
    }
    
    user_prompt = f"""Given this cipher spec and metrics, propose an ImprovementPatch.
CipherSpec JSON:
{json.dumps(spec, indent=2)}
Metrics JSON:
{json.dumps(metrics, indent=2)}

The avalanche score is low (ideal is ~0.5). Suggest improvements."""

    new_rounds = random.choice([12, 14, 16])
    
    patch = {
        "summary": f"Improve {algo}-style cipher diffusion and nonlinearity for better avalanche.",
        "rationale": [
            f"Identity components provide minimal confusion/diffusion in {arch} structure.",
            f"Increasing rounds from 8 to {new_rounds} improves bit mixing.",
            "Using cryptographically designed S-boxes increases nonlinearity."
        ],
        "new_rounds": new_rounds,
        "replace_components": improved_components,
        "add_notes": f"Patched to improve avalanche. Current: {low_avalanche:.2f}, target: ~0.5. Not a security guarantee."
    }
    
    return make_example(
        SYSTEM_PROMPTS["patch"],
        user_prompt,
        json.dumps(patch, indent=2)
    )


def gen_code_example(algo: str, seed: int) -> Dict:
    """Generate a Python code example with comments."""
    info = ALGORITHMS[algo]
    arch = info["arch"]
    block = info["block"]
    key = random.choice(info["key"])
    rounds = random.choice(info["rounds"])
    
    if arch == "SPN":
        components = {"sbox": "sbox.aes", "perm": "perm.aes_shiftrows", "linear": "linear.aes_mixcolumns", "key_schedule": "ks.sha256_kdf"}
    elif arch == "FEISTEL":
        components = {"f_sbox": "sbox.aes", "f_perm": "perm.identity", "key_schedule": "ks.sha256_kdf"}
    else:
        components = {"arx_add": "arx.add_mod32", "arx_rotate": "arx.rotate_left_5", "key_schedule": "ks.sha256_kdf"}
    
    name = f"Auto{algo}_{seed}"
    
    spec = {
        "name": name,
        "architecture": arch,
        "block_size_bits": block,
        "key_size_bits": key,
        "rounds": rounds,
        "components": components,
        "version": "0.2",
        "notes": f"Auto-generated {algo}-style cipher",
        "seed": seed
    }
    
    user_prompt = f"""Write Python code that:
1. Creates this CipherSpec
2. Exports a standalone cipher module to '{name}_cipher.py'
3. Runs the self_test() function to verify correctness

Spec:
{json.dumps(spec, indent=2)}

Include comments explaining each step."""

    code = f'''"""
Auto-generated {algo}-style block cipher ({arch} architecture).

This script:
1. Creates a CipherSpec for {name}
2. Exports the cipher as a standalone Python module
3. Runs the self_test() to verify encrypt/decrypt roundtrip

Research/education only. NOT for production use.
"""

from cipherlab.cipher.spec import CipherSpec
from cipherlab.cipher.exporter import export_cipher_module

# Step 1: Define the cipher specification
# Using {arch} architecture with {block}-bit blocks and {key}-bit keys
spec = CipherSpec(
    name="{name}",
    architecture="{arch}",
    block_size_bits={block},
    key_size_bits={key},
    rounds={rounds},
    components={json.dumps(components)},
    version="0.2",
    notes="{algo}-style cipher for research",
    seed={seed}
)

# Step 2: Export the cipher as a standalone Python module
# This generates complete encrypt/decrypt functions with all components inlined
code = export_cipher_module(spec)

# Step 3: Write the generated code to a file
out_path = "{name}_cipher.py"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(code)
print(f"Exported cipher module to: {{out_path}}")

# Step 4: Verify correctness by running self_test()
# This tests that decrypt(encrypt(plaintext)) == plaintext
ns = {{}}
exec(compile(code, out_path, "exec"), ns, ns)
ns["self_test"]()
print("Self-test passed! Cipher roundtrip is correct.")
'''
    
    return make_example(
        SYSTEM_PROMPTS["code"],
        user_prompt,
        code
    )


def gen_summary_example(algo: str) -> Dict:
    """Generate a KB summary example for an algorithm."""
    info = ALGORITHMS[algo]
    arch = info["arch"]
    block = info["block"]
    keys = "/".join(str(k) for k in info["key"])
    rounds = "/".join(str(r) for r in info["rounds"])
    
    component_notes = {
        "SPN": "S-box substitution, ShiftRows permutation, MixColumns diffusion",
        "FEISTEL": "Feistel F-function with S-box, key mixing, swap halves",
        "ARX": "Add-Rotate-XOR operations, modular arithmetic",
    }
    
    user_prompt = f"Give me a high-level summary of {algo} for a block-cipher knowledge base. Include: architecture, block size, key size, rounds, main components, and practical caveats."
    
    summary = f"""# {algo}

## High-level Summary
- **Architecture:** {arch}
- **Block size:** {block} bits
- **Key sizes:** {keys} bits
- **Rounds:** {rounds}

## Core Structure and Components
- {component_notes[arch]}
- Key schedule derives round keys from master key
- {'Input/output whitening (XOR with key)' if algo in ['Twofish', 'IDEA'] else 'Standard round key addition'}

## Practical Notes / Caveats
- {'Widely standardized (NIST FIPS 197)' if algo == 'AES' else 'Designed for research/comparison purposes'}
- {f'{block}-bit blocks have birthday-bound limits (~2^32 blocks)' if block == 64 else f'{block}-bit blocks provide adequate security margin'}
- **Research only**: This implementation is for education, not production security.

## Relevance for Component-Based Construction
- Provides reference patterns for {arch} cipher design
- Useful for comparing S-box choices, diffusion strategies, and key schedules
"""
    
    return make_example(
        SYSTEM_PROMPTS["tutor"],
        user_prompt,
        summary.strip()
    )


def gen_comparison_example(algo1: str, algo2: str) -> Dict:
    """Generate a comparison example between two algorithms."""
    info1 = ALGORITHMS[algo1]
    info2 = ALGORITHMS[algo2]
    
    user_prompt = f"Compare {algo1} to {algo2} for a study note. Focus on architecture differences and what that means for component selection. Keep it under 300 words."
    
    comparison = f"""{algo1} vs {algo2} (Study Note)

**Architecture:**
- {algo1}: {info1['arch']} ({info1['block']}-bit blocks)
- {algo2}: {info2['arch']} ({info2['block']}-bit blocks)

**Key Differences:**

1. **Invertibility Requirements:**
   {'SPN requires invertible S-box, permutation, and linear layers.' if info1['arch'] == 'SPN' else ''}
   {'Feistel structure automatically provides invertibility via the Feistel swap.' if info1['arch'] == 'FEISTEL' else ''}
   {'ARX uses modular arithmetic (addition, rotation, XOR) which is self-inverting.' if info1['arch'] == 'ARX' else ''}

2. **Block Size Implications:**
   - {algo1}: {info1['block']}-bit blocks {'(birthday-bound concerns for high-volume use)' if info1['block'] == 64 else '(adequate for modern security requirements)'}
   - {algo2}: {info2['block']}-bit blocks {'(birthday-bound concerns for high-volume use)' if info2['block'] == 64 else '(adequate for modern security requirements)'}

3. **Component Selection:**
   - S-box choice affects nonlinearity and differential/linear cryptanalysis resistance
   - Linear diffusion layer determines how quickly bit changes propagate
   - Round count trades off security margin vs. performance

**For Component-Based Generation:**
When choosing between {algo1}-style and {algo2}-style designs:
- Prefer {algo1} when: {info1['arch']} properties are desirable
- Prefer {algo2} when: {info2['arch']} properties are desirable

Both provide valid reference patterns for the Crypto Cipher Lab component framework.
"""
    
    return make_example(
        SYSTEM_PROMPTS["tutor"],
        user_prompt,
        comparison.strip()
    )


def gen_qa_example(seed: int) -> Dict:
    """Generate a Q&A example about block cipher concepts."""
    questions = [
        ("What is the avalanche effect and why is it important?",
         """The avalanche effect measures how much output changes when a single input bit is flipped.

**Ideal Behavior:** ~50% of output bits should change (mean avalanche ≈ 0.5)

**Why It Matters:**
1. **Confusion:** Small input changes should produce unpredictable output changes
2. **Security:** Prevents attackers from learning about the key or plaintext from small variations
3. **Diffusion:** Each input bit should influence many output bits

**In Crypto Cipher Lab:**
- We measure plaintext avalanche (flip 1 plaintext bit) and key avalanche (flip 1 key bit)
- Good ciphers achieve 0.45-0.55 after sufficient rounds
- Identity components (sbox.identity, perm.identity) produce poor avalanche (~0.0-0.3)

**Improving Avalanche:**
1. Use nonlinear S-boxes (sbox.aes, sbox.serpent)
2. Add strong linear diffusion (linear.aes_mixcolumns, linear.twofish_mds)
3. Increase round count
"""),
        ("What's the difference between SPN and Feistel architectures?",
         """**SPN (Substitution-Permutation Network):**
- Used by: AES, Serpent
- Structure: S-box → Permutation → Linear diffusion → Round key (repeat)
- Invertibility: Requires all components to be invertible
- Example flow: SubBytes → ShiftRows → MixColumns → AddRoundKey

**Feistel Network:**
- Used by: DES, Blowfish, Twofish, Camellia
- Structure: Split block in half, L[i+1] = R[i], R[i+1] = L[i] XOR F(R[i], K[i])
- Invertibility: Automatic! F function doesn't need to be invertible
- Decryption: Use round keys in reverse order

**Key Tradeoffs:**
| Aspect | SPN | Feistel |
|--------|-----|---------|
| Parallelism | High (all bytes processed) | Lower (half-block at a time) |
| Invertibility | Must design carefully | Guaranteed by structure |
| Diffusion speed | Fast (full block) | Slower (half block per round) |
| Round count | Typically 10-14 | Typically 16-32 |

**In Crypto Cipher Lab:** Both are supported via the `architecture` field in CipherSpec.
"""),
        ("How do ARX ciphers differ from S-box based ciphers?",
         """**ARX (Add-Rotate-XOR) Ciphers:**
- Used by: RC5, RC6, ChaCha, IDEA
- Operations: Modular addition, bitwise rotation, XOR
- No lookup tables (S-boxes)

**Key Characteristics:**

1. **No S-boxes:** Security comes from mixing addition (nonlinear in binary) with XOR and rotation
2. **Constant-time:** No table lookups = resistant to cache-timing attacks
3. **Efficient on CPUs:** Uses native CPU instructions
4. **Data-dependent rotations:** RC5/RC6 rotate by amounts derived from the data itself

**Comparison:**

| Feature | S-box Based | ARX |
|---------|-------------|-----|
| Nonlinearity source | Lookup tables | Modular addition |
| Timing attacks | Vulnerable (cache) | Resistant |
| Hardware efficiency | Moderate | Good |
| Diffusion | Via linear layer | Via rotation/mixing |

**In Crypto Cipher Lab:**
- ARX components: `arx.add_mod32`, `arx.rotate_left_5`, `arx.mul_mod16`
- IDEA uses multiplication mod 2^16+1 for nonlinearity
- RC5/RC6 use data-dependent rotations

Note: ARX analysis is harder due to mixing arithmetic and bitwise operations.
"""),
        ("What makes a good S-box?",
         """A good S-box provides confusion by being highly nonlinear.

**Key Properties:**

1. **High Nonlinearity:** Maximally different from any linear function
   - Measured by Walsh-Hadamard transform
   - AES S-box nonlinearity: 112 (max possible for 8-bit: 120)

2. **Low Differential Uniformity:** Resistant to differential cryptanalysis
   - For input difference Δx, output difference Δy should be unpredictable
   - AES S-box: differential uniformity = 4 (very good)

3. **Low Correlation Immunity:** Resistant to linear cryptanalysis
   - No strong linear approximations between input/output bits
   - AES S-box: LAT max = 16 (very good)

4. **No Fixed Points:** S[x] ≠ x for most x
   - Prevents trivial patterns

**S-boxes in Crypto Cipher Lab:**

| Component | Size | Notes |
|-----------|------|-------|
| `sbox.aes` | 8×8 | Affine transformation over GF(2^8) inverse |
| `sbox.des` | 6×4 | DES S1-S8, not bijective |
| `sbox.serpent` | 4×4 | Eight different 4-bit S-boxes |
| `sbox.blowfish` | 8×32 | Key-dependent, generated from pi digits |

**For Research:** The `sbox.identity` (S[x] = x) is useful as a control to measure how much S-boxes contribute to avalanche.
"""),
    ]
    
    q, a = random.choice(questions)
    
    return make_example(
        SYSTEM_PROMPTS["qa"],
        q,
        a.strip()
    )


def generate_dataset(n_train: int, n_valid: int, seed: int = 42):
    """Generate balanced training and validation datasets."""
    random.seed(seed)
    
    examples = []
    algos = list(ALGORITHMS.keys())
    
    # Target distribution (roughly equal across types)
    n_spec = n_train // 4
    n_patch = n_train // 4
    n_code = n_train // 4
    n_tutor = n_train - n_spec - n_patch - n_code
    
    print(f"Generating {n_train} training examples...")
    print(f"  - CipherSpec: {n_spec}")
    print(f"  - ImprovementPatch: {n_patch}")
    print(f"  - Code: {n_code}")
    print(f"  - Tutor/QA: {n_tutor}")
    
    # Generate CipherSpec examples (one per algorithm, then random)
    for i, algo in enumerate(algos):
        examples.append(gen_cipherspec_example(algo, i))
    for i in range(n_spec - len(algos)):
        examples.append(gen_cipherspec_example(random.choice(algos), 100 + i))
    
    # Generate ImprovementPatch examples
    for i, algo in enumerate(algos):
        examples.append(gen_improvement_example(algo, i))
    for i in range(n_patch - len(algos)):
        examples.append(gen_improvement_example(random.choice(algos), 200 + i))
    
    # Generate Code examples
    for i, algo in enumerate(algos):
        examples.append(gen_code_example(algo, i))
    for i in range(n_code - len(algos)):
        examples.append(gen_code_example(random.choice(algos), 300 + i))
    
    # Generate Tutor/QA examples
    # Summaries for each algorithm
    for algo in algos:
        examples.append(gen_summary_example(algo))
    
    # Comparisons
    for i in range(20):
        a1, a2 = random.sample(algos, 2)
        examples.append(gen_comparison_example(a1, a2))
    
    # Q&A examples
    for i in range(n_tutor - len(algos) - 20):
        examples.append(gen_qa_example(400 + i))
    
    # Shuffle
    random.shuffle(examples)
    
    # Split
    train_examples = examples[:n_train]
    valid_examples = []
    
    # Generate validation set (different seeds)
    random.seed(seed + 1000)
    for i in range(n_valid // 4):
        valid_examples.append(gen_cipherspec_example(random.choice(algos), 1000 + i))
        valid_examples.append(gen_improvement_example(random.choice(algos), 1100 + i))
        valid_examples.append(gen_code_example(random.choice(algos), 1200 + i))
        valid_examples.append(gen_qa_example(1300 + i))
    
    # Shuffle validation
    random.shuffle(valid_examples)
    valid_examples = valid_examples[:n_valid]
    
    return train_examples, valid_examples


def save_jsonl(examples: List[Dict], path: Path):
    """Save examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def main():
    print("=" * 60)
    print("Block Cipher SFT Dataset Generator")
    print("=" * 60)
    
    train, valid = generate_dataset(TRAIN_SIZE, VALID_SIZE)
    
    save_jsonl(train, TRAIN_FILE)
    save_jsonl(valid, VALID_FILE)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(valid)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
