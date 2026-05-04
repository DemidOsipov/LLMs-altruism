# Social Distance and Altruistic Behavior in Large Language Models

Code and data for the master's thesis:

> **Social Distance and Altruistic Behavior in Large Language Models**  
> Demid Osipov, New Economic School, 2026

## Overview

This repository contains the experiment code and results for a systematic study of altruism in two frontier LLMs — **DeepSeek V3.2** and **Grok 4.1 Fast** — using the Dictator Game. Models are presented with 1,581 conditions spanning 51 social distance levels (rank 1–100) and 31 log-spaced stake sizes ($1–$250,000), with each condition repeated 3 times. Results are compared to the human benchmark from Bechler et al. (2015).

Both models show a significant negative social-distance effect and a significant negative stake effect, but differ in magnitude and shape. Both are more generous than humans at intermediate social distances; at extreme distance only DeepSeek remains significantly more generous, while Grok is statistically indistinguishable from humans.

## Repository structure

```
run_v2_main.py     — Main experiment: DeepSeek V3.2 and Grok 4.1 Fast
run_v2_robust.py   — Robustness checks (temperature × 2, alternative prompt)
analyze_models.py  — Generates all figures

out/
  results_v2_deepseek.csv          — Main experiment, DeepSeek V3.2 (4,743 rows)
  results_v2_grok.csv              — Main experiment, Grok 4.1 Fast (4,743 rows)
  results_v2_robust_deepseek_*.csv — Robustness conditions, DeepSeek
  results_v2_robust_grok_*.csv     — Robustness conditions, Grok
  humans_digitized.csv             — Human benchmark (Bechler et al. 2015, digitized)
  fig_*.png                        — Figures
  numbers_for_tex.txt              — Key regression numbers
  analysis_v2_output.txt           — Full analysis output
```

## Reproducing the results

**Requirements:** Python 3.10+, packages: `openai`, `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`

**API access:** Both models are queried via [OpenRouter](https://openrouter.ai). Set your API key:
```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

**Run the main experiment** (≈9,500 API calls total for both models):
```bash
python run_v2_main.py
```

**Run robustness checks** (≈1,200 API calls per model):
```bash
python run_v2_robust.py
```

**Regenerate figures:**
```bash
python analyze_models.py
```

## Key results

| Model | $\hat{c}_\text{SD}$ | $\hat{c}_{\log\text{stake}}$ | $R^2$ |
|---|---|---|---|
| DeepSeek V3.2 | −0.00292*** | −0.00709*** | 0.54 |
| Grok 4.1 Fast | −0.00572*** | −0.01019*** | 0.86 |

At SD=20, both models give ~38–39% on average vs. 11.5% for humans ($p < 0.001$).  
At SD=100, DeepSeek gives 12.2% vs. 6.5% for humans ($p = 0.003$); Grok gives 5.5% ($p = 0.52$).

## Reference

Bechler, C., Green, L., & Myerson, J. (2015). Proportion offered in the Dictator and Ultimatum Games decreases with amount and social distance. *Behavioural Processes, 115*, 149–155.
