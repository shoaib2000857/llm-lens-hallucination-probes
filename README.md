# Internal-State Hallucination Detection: Code and Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24081563.svg)](https://doi.org/10.5281/zenodo.24081563)

**LLM Lens, Keshav Memorial Institute of Technology, Hyderabad**

Code, configurations and released data for:

- **Reading Hallucination from the Inside: Supervision, Architecture, Generalization, Human Validity and Model Change in Internal-State Detectors** (arXiv, 2026), *link to be added*
- **Distilling judge supervision into internal-state hallucination detectors: generalization, inductive bias and human validity** (under review), *link to be added*
- **Weakly Supervised Distillation of Hallucination Signals into Transformer Representations**: [arXiv:2604.06277](https://arxiv.org/abs/2604.06277)

We train probes on the hidden states of Qwen3-8B and Ministral-3-8B to detect answers containing claims that are unsupported by a supplied passage. The probes are supervised offline by a panel of three LLM judges and evaluated across five QA datasets, against 480 blinded human references, and under model replacement and fine-tuning.

## Repository layout

| Path | Contents |
|---|---|
| `src/portability/` | Hidden-state extraction, probe architectures (M0–M4, linear/MLP controls, ablations), training and evaluation |
| `scripts/` | Entry points for every experiment stage (generation, judging, probe training, transfer, LoRA updates, recovery, alignment, baselines) |
| `configs/` | Experiment configurations with pinned model and dataset revisions |
| `tests/` | Unit tests |
| `analysis/` | Scripts behind the papers' statistics (learning curves, paired contrasts, transfer, human references, audits) |
| `paper_figures/` | Figure and table generation scripts and the rendered figures |
| `data/responses/` | Generated answers with questions and gold answers, one `jsonl.gz` per corpus (source passages are not redistributed; see `DATA_LICENSE.md`) |
| `data/labels/` | Per-response judge verdicts, consensus labels and token-confidence summaries, one CSV per corpus |
| `data/human/` | 480 de-identified human reference labels and agreement tables |
| `data/predictions/` | Output probabilities of every reported probe fit (`fit, sample_id, label, probability, fold`) |
| `results/` | Derived metric tables used in the papers |

Corpus names follow `stage1_<corpus>`: source tests (`qwen3_8b`, `ministral3_8b`), distribution shifts (`distribution_v1_*`, `distribution_v2_*`), same-family targets (`samefamily_v2_*`) and LoRA checkpoints (`updates_v1*`).

**Not included:**
- Raw hidden-state tensors (about 1 TB). Regenerate them with `scripts/extract_representations.py` and the configs.
- Trained probe weights: *Hugging Face link to be added*.

## Reproducing the analyses

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt     # exact versions used on the research server (CUDA build)
python -m pytest tests
```

The analysis scripts in `analysis/` read saved predictions and labels. Edit the paths at the top of `analysis/lib.py` to point to `data/` and `results/`.

## Licences

- Code: MIT (`LICENSE`).
- Labels, human references, predictions and results: CC BY 4.0.
- Response files: CC BY-SA 4.0, inherited from the source datasets (`DATA_LICENSE.md`).

## Citation

Code and data: Salehmohamed et al. (2026), *Internal-State Hallucination Detection: Code and Data*, Zenodo, https://doi.org/10.5281/zenodo.24081563 (always resolves to the latest version). For the papers, see `CITATION.cff`.
