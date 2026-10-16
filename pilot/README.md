# Paper 2 pilot v2 - boundary judgements

Send each annotator ONE workbook and nothing else:
- `pilot_annotator_A.xlsx` -> annotator 1
- `pilot_annotator_B.xlsx` -> annotator 2

Same 200 rows, different order. Instructions are on the first sheet; they fill only the yellow Answer column.
About 2-3 hours each. Don't tell them what is being tested - only "we are checking where words should be split".

Do NOT send anything in `_key/` (it says which rows are which group).

When both come back:
    python analyse_pilot.py pilot_annotator_A.xlsx pilot_annotator_B.xlsx _key/pilot_key_DO_NOT_SEND.csv

`prereg_pilot.json` (v2) fixes the three predictions BEFORE any answers exist:
  P1 convention differences inside compounds are principled alternatives (experts say "either")
  P2 prefix/negation differences are unstable (experts contradict each other)
  P3 the original claim (disagreement = contested), kept for honesty; the corpus proxy predicts it fails
Don't edit it after answers arrive. v1 was replaced before any annotation, after the corpus-proxy test
(`h2c_proxy_summary.py`). `build_pilot.py` / `make_workbooks.py` regenerate everything (seeded).
