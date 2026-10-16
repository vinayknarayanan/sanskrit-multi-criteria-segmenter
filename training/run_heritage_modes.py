#!/usr/bin/env python3
"""Kill test 1: re-run the Sanskrit Heritage segmenter in BOTH modes on the 3,259-item sample.

  parse mode   ('mode': 'p')  -- what heritage.py's get_analysis() uses by default; PRUNES solutions
  tagging mode ('mode': 't')  -- "Tagging does not prune any solutions" (heritage.py source) ~ the Complete output

Hypothesis: the old results_inria.csv (median 1 candidate) came from parse mode. If tagging mode returns many
candidates per item, INRIA's 0.474 was an output-mode artefact of the wrapper, not the engine's ceiling.

Needs the INRIA site (blocked from the Claude session) or a local Heritage Platform install. Run on YOUR machine:
  pip install heritage indic-transliteration pandas
  python run_heritage_modes.py --eval shared_eval_sample.csv --out heritage_modes.csv --limit 5      # test first
  python run_heritage_modes.py --eval shared_eval_sample.csv --out heritage_modes.csv                # full run
  # much faster with a local install (https://sanskrit.inria.fr/manual.html#installation):
  python run_heritage_modes.py ... --base-dir ~/Heritage_Platform --method shell
Resumable: rows already in --out are skipped. The web mirror is slow (seconds per item): be polite, keep --sleep.

Output columns: compound, gold_split, split_count, mode, n_solutions, candidates_json (list of piece lists, IAST,
solution order as returned), error. Score with score_heritage.py (top-1 and oracle-of-n).
"""
import argparse, json, multiprocessing as mp, os, time, shutil, pandas as pd
import requests
from indic_transliteration import sanscript
from heritage import HeritagePlatform
from heritage.heritage import HeritageOutput

def solutions(hp, text_deva, mode):
    opts = {"lex": hp.get_lexicon(), "cache": "t", "st": "t", "us": "f", "cp": "t", "t": hp.get_option("t"),
            "mode": mode, "font": "roma", "topic": "", "corpmode": "", "corpdir": "", "sentno": "",
            "text": hp.prepare_input(text_deva)}
    res = hp.get_result("reader", opts)
    if res is None: return []
    an = HeritageOutput(res).extract_analysis(meta=False, structured=False) or {}
    out = []
    for sid in sorted(an, key=lambda k: int(k) if str(k).isdigit() else 0):
        sol = an[sid]; words = sol.get("words", []) if isinstance(sol, dict) else []
        pieces = []
        for w in words:                       # each entry: list of alternative analyses of one segment
            first = w[0] if isinstance(w, list) and w else w
            txt = (first.get("text") or first.get("word") or first.get("form") or "") if isinstance(first, dict) else str(first)
            if txt: pieces.append(sanscript.transliterate(txt, "devanagari", "iast") if any("ऀ" <= ch <= "ॿ" for ch in txt) else txt)
        if pieces: out.append(pieces)
    return out

_HP = None
def _init(base_dir, method, timeout):
    """runs inside the worker process: patch the request timeout, build the platform"""
    global _HP
    import heritage.heritage as _hh
    _get = requests.get
    def _long_get(url, *args, **kw):
        kw["timeout"] = timeout
        return _get(url, *args, **kw)
    _hh.requests.get = _long_get
    _HP = HeritagePlatform(base_dir=base_dir, method=method)

def _work(text_deva, mode):
    return solutions(_HP, text_deva, mode)

def _pool(a):
    return mp.Pool(1, _init, (a.base_dir, a.method, a.timeout))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="web"); ap.add_argument("--base-dir", default="")
    ap.add_argument("--modes", default="p,t"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=90, help="seconds per request (the heritage package default is 10)")
    ap.add_argument("--hard-timeout", type=float, default=240, help="wall-clock seconds per item incl. parsing; stuck items are skipped and marked hard_timeout")
    ap.add_argument("--retry-zero", action="store_true", help="re-query rows that came back with 0 solutions (mostly silent timeouts)")
    a = ap.parse_args()
    pool = _pool(a)
    ev = pd.read_csv(a.eval)
    if a.limit: ev = ev.head(a.limit)
    done = set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        if a.retry_zero:
            shutil.copy(a.out, a.out + ".bak")
            zero = (prev.n_solutions.fillna(0) == 0) & ~prev.error.fillna("").str.startswith("hard_timeout")
            print(f"retrying {zero.sum()} zero-solution rows (backup: {a.out}.bak)")
            prev = prev[~zero]; prev.to_csv(a.out, index=False)
        done = set(zip(prev.compound, prev["mode"]))
    first = not os.path.exists(a.out)
    for i, r in enumerate(ev.itertuples()):
        for mode in a.modes.split(","):
            if (r.compound, mode) in done: continue
            row = dict(compound=r.compound, gold_split=r.gold_split, split_count=r.split_count, mode=mode)
            try:
                job = pool.apply_async(_work, (sanscript.transliterate(r.compound, "iast", "devanagari"), mode))
                sols = job.get(timeout=a.hard_timeout)
                row.update(n_solutions=len(sols), candidates_json=json.dumps(sols, ensure_ascii=False), error="")
            except mp.TimeoutError:
                print(f"\n  hard timeout ({a.hard_timeout:.0f}s) on item {i} [{mode}] {r.compound[:40]} -- skipped")
                pool.terminate(); pool.join(); pool = _pool(a)
                row.update(n_solutions=0, candidates_json="[]", error=f"hard_timeout_{a.hard_timeout:.0f}s")
            except Exception as e:
                row.update(n_solutions=0, candidates_json="[]", error=repr(e)[:300])
            pd.DataFrame([row]).to_csv(a.out, mode="a", header=first, index=False); first = False
            time.sleep(a.sleep if a.method == "web" else 0)
        print(f"{i + 1}/{len(ev)}", end="\r")
    pool.terminate()
    d = pd.read_csv(a.out)
    print("\n", d.groupby("mode").n_solutions.describe().round(1).to_string())
    print("errors:", (d.error.fillna("") != "").sum())
    if a.limit:
        print("\nCHECK the first rows by eye before the full run:")
        print(d[["compound", "mode", "n_solutions", "candidates_json"]].head(2 * a.limit).to_string()[:3000])

if __name__ == "__main__":
    main()
