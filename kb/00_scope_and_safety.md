# Scope, safety, and research framing

This project is designed for **PhD research and education** on *cipher construction workflows* and the use of LLMs as assistants.

## Do not deploy generated ciphers
A cipher that “looks good” under simple heuristics (avalanche, round-trip correctness, etc.) can still be **catastrophically broken**.
Real-world cryptographic security requires:
- years of public analysis,
- formal security arguments (where applicable),
- extensive cryptanalysis (differential, linear, algebraic, integral, related-key, side-channel considerations),
- secure implementations (constant-time, masking, hardened key schedule handling).

## What is reasonable to claim in a thesis
It is reasonable to claim:
- the system generates *syntactically valid* cipher constructions,
- the system enforces structural constraints (invertibility, block-size compatibility),
- the system can suggest plausible improvements and run automated tests,
- the system supports reproducible evaluation and ablation experiments.

It is not reasonable to claim:
- the system “creates secure ciphers” for production,
- a generated cipher is “stronger than AES” (without extraordinary evidence).

## Ethical and defensive framing
Position the work as:
- a **design-space exploration** and **research automation** tool,
- an assistant that helps researchers generate candidates and perform *screening tests*,
- a tool for learning and for producing *testable* cipher definitions.
