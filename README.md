# Change the Answer Key, Change the Winner — code and data

Code, evaluation splits, model predictions and intermediate results for the paper
*Change the Answer Key, Change the Winner: A Convention-Switchable Model for Sanskrit Word Segmentation* (ISCLS 2027 submission).

## Layout

| Folder | What is in it |
|---|---|
| `training/` | `finetune_byt5_frontiers.py` (trains the Frontiers-only and the convention-tagged ByT5-small, and predicts), `run_heritage_modes.py` (Heritage baseline), `configs/` (the `config.json` written by every run reported in the paper) |
| `data/` | Evaluation splits: `shared_eval_sample.csv` (3,259 Frontiers items), `dcs_holdout_eval.csv` (3,000 DCS items, 600 per piece count), `dcs_frequent_eval.csv` (forms DCS annotates 5+ times, §5.4); sandhi rule tables |
| `data/third_party/` | Frontiers corpus (`frontiers_dataset.csv`) and the tagged training file (`dataset_tagged.csv`, 87,595 Frontiers + 87,595 DCS rows). **Check the licences before making this public.** |
| `predictions/` | Model outputs, one folder per run (`results_fr.csv` / `results_dcs.csv` = the two tag settings) |
| `analysis/` | All scoring and comparison scripts; `paths.py` holds every external location |
| `results/` | Outputs of the analysis scripts as used in the paper, plus the baseline system outputs (`results_byt5.csv` = ByT5-Sanskrit, `results_rcnn.csv`, `heritage_modes.csv`) |
| `tools/` | Dataset download and the two checks reported in the paper |
| `pilot/` | The planned expert pilot (design, preregistration, workbooks). The answer key is deliberately not included. |

## Runs reported in the paper

| Run folder | Model | Seed | Epochs | Evaluated on |
|---|---|---|---|---|
| `byt5_frontiers_v2` | Frontiers-only ByT5-small | 13 | 3 | Frontiers eval |
| `byt5_tagged` | tagged ByT5-small | 13 | 2 | Frontiers eval, both tags |
| `byt5_tagged_s2` | tagged ByT5-small | 42 | 2 | Frontiers eval, both tags |
| `tagged_on_dcs`, `tagged_s2_on_dcs` | the two tagged models above | – | – | DCS held-out, both tags |
| `tagged_on_dcsfreq` | tagged, seed 13 | – | – | frequent DCS forms (§5.4) |
| `tagged_s7`, `tagged_s21`, `tagged_s99` | tagged ByT5-small | 7, 21, 99 | 2 | Frontiers eval, both tags |
| `tagged_s{7,21,99}_on_dcs`, `tagged_s{7,21,99}_on_dcsfreq` | the three tagged models above | – | – | DCS held-out / frequent DCS forms, both tags |
| `frontiers_s7`, `frontiers_s21`, `frontiers_s42`, `frontiers_s99` | Frontiers-only ByT5-small | 7, 21, 42, 99 | 3 | Frontiers eval |
| `frontiers_2ep_s13` | Frontiers-only ByT5-small (epoch control) | 13 | 2 | Frontiers eval |
| `frontiers_s{13,7,21,42,99}_on_dcs` | the five Frontiers-only models | – | – | DCS held-out (authentic-key reversal) |

`analysis/five_seed.py` aggregates the five tagged and five Frontiers-only seeds (tag flip, pre-specified ±0.01 equivalence test, epoch control, authentic-key reversal). Model checkpoints are not included.

## External data (not redistributed; put under `external/` or point the `S2_*` variables in `analysis/paths.py` at them)

- Frontiers: `data/third_party/frontiers_dataset.csv` is byte-identical to `dataset.csv` in https://github.com/IrfanAliBabar/frontier_2026 (commit 5811e04, 90,893 rows; MD5 1d7435942c0a9850343ae23c29d2c8e6). It is not the 89,656-row cleaned version the article describes.
- DCS: `git clone https://github.com/OliverHellwig/sanskrit external/sanskrit` (paper used commit 8aeed5a)
- SaCTI / UoH: `git clone https://github.com/ashishgupta2598/SaCTI external/SaCTI` (commit fecb9ef)
- vidyut data release py-0.4.0 (`sandhi/rules.csv`) into `external/vidyut-data/`
- SIGHUM, Hackathon: `python tools/download_conventions.py` (Hugging Face, user `chronbmm`). Commits checked in the paper:
  sighum 25f3e07, sighum2 c6110cb, sighum3 daf0b40, sighum6 0b46c41, sighum7 9e089bc, sighum-filtered c52359e,
  sighum-filtered-new 50abead, sandhi-split-2018 9dacbc8
- SandhiKosh (Bhardwaj et al. 2018) `.xls` files into `external/sandhikosh/`

## How to run

```
pip install -r requirements.txt
cd results
python ../analysis/dcs_surface_index.py      # builds cache/dcs_surface.pkl from DCS (once)
python ../analysis/dcs_index.py              # builds cache/dcs_index.pkl (once; needed by rescore.py)
python ../analysis/robust_ci.py              # Tables 2 and 5, tag-flip and baseline intervals
```

Scripts read their inputs from the current folder, so run them from `results/`.
`robust_ci.py` has been re-run from this layout and reproduces the paper's figures (e.g. mirror gap +0.190 [0.166, 0.212], `<fr>` minus Frontiers-only 0.000 [−0.012, +0.013]).

## Script per paper section

| Section | Scripts |
|---|---|
| §3 Data | `build_dcs_pairs.py` (DCS token → division pairs), `extract_units.py` (SIGHUM/Hackathon verses → word units), `full_data.py` (population weights → `full_piece_distribution.csv`) |
| §4 Model and baselines | `training/finetune_byt5_frontiers.py`, `training/run_heritage_modes.py` |
| §4 Scoring | `rescore.py` + `derive_engine.py` (strict F1), `robust_ci.py` (form-agnostic F1, bootstrap), `sandhi_fp_check.py` (lenient variants), `seed2.py`, `score_dcs_holdout.py`, `score_heritage_w.py`, `pooled_strict.py` (Table 2, pooled strict and form-agnostic F1), `pooled.py` / `pooled_dcs.py` |
| §5.1 Same words, both keys | `corpus_compare.py` (Frontiers pieces vs DCS), `converted_gold.py` (Table 4), `validate_key.py` |
| §5.2 One model | `robust_ci.py`, `score_heritage_w.py` |
| §5.3 Dataset map, UoH | `convention_distance.py`, `convention_distance_tol.py` (zero-tolerance footnote, `TOL=0`), `uoh_vs_dcs.py`, `uoh_convention_study.py`, `frontiers_vs_uoh.py`, `sandhikosh_compare.py` |
| §5.4 Consistency | `consistency_proxy.py`, `consistency_summary.py` |
| §5.5 Benchmark problems | `tools/check_sighum_splits.py` |
| Method, leakage | `tools/check_training_overlap.py` |

## Known gaps

- The sampling that produced `dataset_tagged.csv`, `dcs_holdout_eval.csv` and `dcs_frequent_eval.csv` was not saved as a script; the files themselves are included.
- `results_byt5.csv` (ByT5-Sanskrit) and `results_rcnn.csv` are system outputs produced earlier; the commands that generated them are not included.
- A few paper figures were computed interactively and have no standalone script yet: the zero-tolerance tag-flip gap (0.229), Table 3, the 547 Frontiers forms with two divisions, and Figure 1.
- Only `robust_ci.py` has been re-run end to end from this layout; `five_seed.py` still uses the analysis machine's paths.
