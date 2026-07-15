---
name: cross-domain-figure-data-source
description: Canonical data source for thesis cross-domain One-For-All figures (avoid stale notebook JSON)
metadata:
  type: project
---

The thesis cross-domain One-For-All figures (`Documentazione/images/grafici/fig_cross_domain_*.pdf`) and the table `tab:cross-domain-results` in `Documentazione/sections/4_results.tex` must both be derived from the canonical per-seed results in `results/cross_domain_one_for_all/cross_domain.csv` (seed 42 = the values quoted in the table).

**Why:** The old plotting cell in `notebooks/nonLinearApproach/approach3OneForAll/croossval.py` reads a *separate, stale* set of JSON files in `notebooks/.../approach3OneForAll/<LLama_Gemma_XXX>/results_metrics/cross_dataset_eval__*.json`. Those JSONs have implausibly high accuracy (~0.97–0.99, evaluated on a much smaller eval set, n≈1604 vs 23054) and are even missing the encoder=BBC scenarios (the `LLama_Gemma_BBC/results_metrics/` folder is empty), so the old figures contradicted the table.

**How to apply:** Regenerate the two Gemma↔Llama cross-domain figures with `python src/plot_cross_domain_seed42_thesis.py` (filters seed 42, writes the exact thesis filenames into `Documentazione/images/grafici/`). Then recompile with `latexmk -lualatex -interaction=nonstopmode main.tex` from `Documentazione/`. The thesis uses LuaLaTeX (the `emoji` package requires it) + natbib. Do NOT regenerate from `croossval.py`.
