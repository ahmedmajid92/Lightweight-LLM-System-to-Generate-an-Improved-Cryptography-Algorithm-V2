# How RAG is used in this system

The assistant receives:
1. Your **current cipher spec** (JSON)
2. Retrieved KB chunks relevant to your query

The assistant should use KB context to:
- explain design principles
- warn about structural pitfalls
- propose compatible component swaps
- recommend evaluations to run

## Good RAG prompts
- “Given my SPN spec, what components should I change to improve diffusion?”
- “Does my key schedule look weak? Suggest improvements.”
- “Which heuristics should I report in my evaluation chapter?”

## Bad prompts
- “Generate a cipher that is unbreakable”
- “Give me a cipher stronger than AES” (unfalsifiable; requires extreme evidence)

## Thesis suggestion
Log:
- user queries
- retrieved KB sources
- model outputs
- spec changes
- evaluation scores

This produces a reproducible research trail, useful for your methodology and results chapters.
