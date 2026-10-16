"""All external locations in one place. Override any of them with an environment variable.
Defaults assume this layout next to the repo root:
  data/         evaluation splits and the training CSVs shipped with the repo
  predictions/  model outputs (results_*.csv), one folder per run, as in predictions/README.md
  external/     downloads you fetch yourself (see README.md): DCS, SaCTI, vidyut data, SandhiKosh, Hugging Face sets
  cache/        large files the scripts build on first run (DCS indexes, .pkl)
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(env, *default): return os.environ.get(env, os.path.join(ROOT, *default))
DATA      = _p('S2_DATA', 'data') + os.sep
RUNS      = _p('S2_RUNS', 'predictions') + os.sep          # was run_kit/runs/
CACHE     = _p('S2_CACHE', 'cache') + os.sep
DCS_CONLLU = _p('S2_DCS', 'external', 'sanskrit', 'dcs', 'data', 'conllu', 'files')   # github.com/OliverHellwig/sanskrit
SACTI     = _p('S2_SACTI', 'external', 'SaCTI', 'data', 'sacti', 'coarse') + os.sep  # github.com/ashishgupta2598/SaCTI
VIDYUT_RULES = _p('S2_VIDYUT', 'external', 'vidyut-data', 'sandhi', 'rules.csv')    # vidyut data release py-0.4.0
SANDHIKOSH = _p('S2_SANDHIKOSH', 'external', 'sandhikosh') + os.sep
HF        = _p('S2_HF', 'external', 'hf') + os.sep                                   # tools/download_conventions.py
os.makedirs(CACHE, exist_ok=True)
