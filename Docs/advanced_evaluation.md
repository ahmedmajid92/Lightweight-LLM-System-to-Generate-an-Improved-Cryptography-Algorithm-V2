# Advanced Evaluation in Crypto Cipher Lab

## Purpose

The **Advanced evaluation** stage is the system's deeper deterministic testing layer. It comes after the simpler local avalanche metrics and is meant to answer a stronger question:

**Is the current cipher design functionally correct, structurally compatible, and showing acceptable heuristic cryptographic behavior?**

In the Streamlit app, this is Section 3:

- `Roundtrip verification`
- `SAC analysis`
- `S-box profiling`
- `I/O compatibility`

Implementation entry point in the UI:

- `app/streamlit_app.py`


## What It Includes

The advanced evaluation stage has four main parts:

1. **Roundtrip verification**
2. **Strict Avalanche Criterion (SAC)**
3. **S-box analysis**
4. **Component I/O compatibility analysis**

These are all deterministic or reproducible checks driven by a seed and explicit test counts.


## 1. Roundtrip Verification

### Goal

Roundtrip verification checks the most basic correctness property of a block cipher:

`P = D(E(P, K), K)`

That means if the system encrypts a plaintext `P` with key `K`, then decrypting the ciphertext with the same key must return the original plaintext exactly.

### How it works

The system:

- builds the current cipher from the selected `CipherSpec`
- generates many random plaintext/key pairs
- encrypts each plaintext
- decrypts the ciphertext
- compares the decrypted result to the original plaintext

### What is recorded

For each run, the system records:

- total number of test vectors
- how many passed
- how many failed
- a small sample of failure details
- elapsed time
- seed

### Why it matters

If roundtrip fails, the design is functionally broken. This is treated as the most serious type of problem in the advanced evaluation pipeline.

### Severity in diagnostics

- Roundtrip failure is treated as **critical**

### Code

- `cipherlab/evaluation/roundtrip.py`


## 2. Strict Avalanche Criterion (SAC)

### Goal

SAC measures diffusion more carefully than the basic avalanche score.

The idea is:

- flip exactly one input bit
- re-encrypt
- observe how many output bits change

For a well-diffused cipher, each output bit should flip with probability close to `0.5`.

### Two SAC checks are run

The system runs SAC twice:

1. **Plaintext SAC**
   - perturb one plaintext bit at a time
   - keep key fixed for that trial

2. **Key SAC**
   - perturb one key bit at a time
   - keep plaintext fixed for that trial

This is useful because a cipher may diffuse plaintext changes reasonably well while still having a weak key schedule, or vice versa.

### How it works

For each input bit position:

- generate random plaintext and key
- encrypt once
- flip the selected bit
- encrypt again
- compute output Hamming distance
- convert that to a fraction of output bits changed
- repeat for many trials

Then the system aggregates results across all bit positions.

### Main reported values

For each SAC run, the system records:

- `global_mean`
  - average output flip probability across all input bits
  - ideal target is around `0.5`
- `global_std`
  - variation across bit positions
  - lower is more uniform
- `min_bit_prob`
  - weakest input bit position
- `max_bit_prob`
  - strongest input bit position
- `sac_deviation`
  - mean absolute distance from `0.5`
  - lower is better

### Pass heuristic

The current code marks SAC as passing when:

- `sac_deviation < 0.05`
- and `min_bit_prob > 0.35`

So a cipher can still fail even if its mean looks acceptable, if some specific bit positions are weak.

### Why it matters

This stage gives more fine-grained evidence than the basic local avalanche score. It helps detect:

- weak diffusion
- uneven diffusion
- key schedule weakness
- specific bad bit positions

### Severity in diagnostics

- SAC weakness is treated as **warning**
- if `global_mean < 0.40`, the system also creates a separate **low avalanche** warning

### Code

- `cipherlab/evaluation/avalanche.py`


## 3. S-box Analysis

### Goal

The system evaluates the cryptographic quality of analyzable S-box components using standard heuristic properties.

It checks:

- `DDT max`
- `LAT max abs`
- bijectivity

### What these mean

#### DDT max

`DDT` means **Difference Distribution Table**.

This is used as a heuristic for resistance to differential cryptanalysis. Lower is better.

#### LAT max abs

`LAT` means **Linear Approximation Table**.

This is used as a heuristic for resistance to linear cryptanalysis. Lower is better.

#### Bijectivity

The system also checks whether the S-box can be inverted correctly.

For SPN-style reversible substitution layers, this matters because decryption depends on invertibility.

### Ratings used by the system

The code assigns simple quality labels:

For **4-bit S-boxes**:

- DDT `<= 4` -> `good`
- DDT `<= 6` -> `fair`
- otherwise -> `poor`

- LAT `<= 4` -> `good`
- LAT `<= 6` -> `fair`
- otherwise -> `poor`

For **8-bit S-boxes**:

- DDT `<= 4` -> `good`
- DDT `<= 8` -> `fair`
- otherwise -> `poor`

- LAT `<= 16` -> `good`
- LAT `<= 32` -> `fair`
- otherwise -> `poor`

### Important implementation nuance

The current advanced evaluation does **not** only analyze the currently selected S-box in the working cipher.

Instead, it calls:

- `analyze_all_sboxes(registry)`

That means it profiles the analyzable S-box components in the active registry, with these exclusions:

- Feistel F-function components are skipped:
  - `sbox.tea_f`
  - `sbox.xtea_f`
  - `sbox.simon_f`
  - `sbox.hight_f`
- `sbox.identity` is skipped

So this part is better understood as **registry-level S-box profiling** rather than only **current-cipher S-box profiling**.

### Severity in diagnostics

S-box issues are treated as **warning** when:

- differential rating is `poor`
- or linearity rating is `poor`
- or the S-box is not bijective

### Code

- `cipherlab/evaluation/sbox_analysis.py`
- `cipherlab/cipher/cryptanalysis.py`


## 4. I/O Compatibility Analysis

### Goal

This checks whether connected cipher components match each other structurally.

Examples:

- output width of one component must match input width of the next
- architecture-specific expectations must hold
- a staged patch should not create incompatible component wiring

### How it is used in the app

In the UI, I/O compatibility is checked with a separate button:

- `Check component I/O compatibility`

So it is shown in the same advanced evaluation section, but it is not stored inside the main `EvaluationReport` object in the same way as roundtrip, SAC, and S-box results.

### Why it matters

This catches structural problems that may not appear as a syntax error but still make a design invalid or unsafe to evolve further.

### Code

The UI calls:

- `detect_mismatches(...)`

This comes from the evolution/mismatch-analysis path used elsewhere in the system as well.


## Diagnostic Layer After Evaluation

After roundtrip, SAC, and S-box analysis are complete, the system converts the results into structured diagnostics.

These diagnostics are later used by the AI feedback and iterative improvement stages.

### Diagnostic rules used now

1. **Roundtrip failures**
   - severity: `critical`
   - focus: encrypt/decrypt implementation

2. **SAC weak bits**
   - severity: `warning`
   - focus:
     - plaintext SAC -> permutation or linear layer
     - key SAC -> key schedule

3. **S-box weaknesses**
   - severity: `warning`
   - focus: S-box replacement

4. **Low overall avalanche**
   - severity: `warning`
   - triggered when SAC mean is below `0.40`

### Code

- `cipherlab/evaluation/feedback.py`


## Inputs Controlled by the User

In the UI, the user can set:

- `Roundtrip test vectors`
  - range: `100` to `10000`
  - default: `1000`

- `SAC trials per bit`
  - range: `50` to `2000`
  - default: `500`

Higher values give more stable measurements but increase runtime.


## Relation to the Basic Local Metrics

The system also has an earlier local metrics stage based on simpler avalanche scoring.

That earlier stage is faster and is mainly used for:

- quick screening
- rough scoring
- early issue flags

Advanced evaluation is different because it adds:

- algebraic correctness checking
- per-bit SAC analysis
- structured S-box profiling
- diagnostic extraction
- compatibility checking

So the basic stage is a **fast heuristic filter**, while advanced evaluation is the **deeper deterministic analysis layer**.


## What the Advanced Evaluation Does Not Prove

Even when advanced evaluation looks good, it does **not** prove that a cipher is secure.

It does not replace:

- formal cryptanalysis
- peer review
- proofs
- standardization-level security analysis
- implementation side-channel review

This stage should be described as:

**deterministic heuristic evaluation for research prototypes**

not as proof of real-world security.


## Summary

In this system, **Advanced evaluation** means:

- verify the cipher actually decrypts what it encrypts
- measure diffusion more rigorously with SAC on plaintext and key bits
- profile S-box quality using DDT, LAT, and bijectivity
- check structural component compatibility
- convert the findings into actionable diagnostics for the next improvement step

This makes it one of the most important stages in the full design-improve-evaluate loop, because it provides the evidence used for both human review and AI-guided refinement.
