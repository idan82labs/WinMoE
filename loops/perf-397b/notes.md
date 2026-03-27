# Quality Frontier Notes

## 2026-03-26 — K=3/4/5 Quality Comparison

### Test prompts
1. **General knowledge**: "What is quantum computing?"
2. **Math reasoning**: "Solve step by step: A train leaves station A at 60 km/h..."
3. **Code generation**: "Write a Python function that checks if a string is a valid IPv4 address."

### Findings

**All three K values produce coherent, factually correct, well-structured output.**

There is NO observable quality cliff between K=3 and K=5 on these prompts.
The outputs are remarkably similar in structure, content, and accuracy:

| Dimension | K=3 | K=4 | K=5 |
|-----------|-----|-----|-----|
| Coherence | Full | Full | Full |
| Factual accuracy | Correct | Correct | Correct |
| Structure | Markdown headings, lists | Same | Same |
| Math reasoning | Correct setup, LaTeX | Same | Same |
| Code quality | Correct impl, edge cases | Not tested | Correct, slightly more thorough docstring |
| Verbosity | Normal | Slightly more | Slightly more |

### Key observation
The quality difference between K=3 and K=5 is **marginal at best** on IQ2_XXS quantization.
This makes sense: at 2-bit quantization, the quantization noise already dominates over the
expert-count noise. Reducing from 5 to 3 experts drops routing probability mass from ~78%
to ~63%, but the 2-bit weight noise is a much larger error source.

**Implication**: At IQ2_XXS, K=3 is defensible. The quality gap would likely be larger at
Q4_K_M quantization where weight precision is higher and expert-count becomes the dominant
quality variable.

### Performance summary

| K | Gen tok/s (avg) | Quality | Verdict |
|---|----------------|---------|---------|
| 3 | 1.6 | Good | **Best speed, acceptable quality** |
| 4 | 1.5 | Good | Marginal quality gain, -6% speed |
| 5 | 1.4 | Good | No visible quality gain over K=4, -12.5% speed |
