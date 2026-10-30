"""ByT5-Sanskrit (chronbmm/sanskrit5-multitask) on the two-key words, using the authors' own inference recipe
(github.com/dharmamitra/byt5-sanskrit-analyzers, applications/segmentation-lemma-tagging/inf/model.py):
prefix "S ", greedy generate, max_length 512, "_" -> space. Also re-runs 200 Frontiers evaluation items so the
outputs can be compared with the earlier results_byt5.csv (checks the recipe is the same one)."""
import os, sys, torch, pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
here = os.path.dirname(os.path.abspath(__file__))
name = 'chronbmm/sanskrit5-multitask'
tok = AutoTokenizer.from_pretrained(name); model = T5ForConditionalGeneration.from_pretrained(name)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model.to(dev).eval(); print('device', dev)
def run(words, bs=32):
    out = []
    for i in range(0, len(words), bs):
        x = tok(['S ' + w for w in words[i:i + bs]], return_tensors='pt', padding=True, truncation=True, max_length=512).to(dev)
        with torch.no_grad(): y = model.generate(**x, max_length=512)
        out += [s.replace('_', ' ').strip() for s in tok.batch_decode(y, skip_special_tokens=True)]
        print(f'{min(i + bs, len(words))}/{len(words)}', end='\r')
    print(); return out
ev = pd.read_csv(os.path.join(here, 'twokey_eval.csv'))
ev['predicted'] = run(ev.compound.tolist())
ev.to_csv(os.path.join(here, 'results_byt5skt_twokey.csv'), index=False)
chk = pd.read_csv(os.path.join(here, '..', 'data', 'shared_eval_sample.csv')).drop_duplicates('compound').sample(200, random_state=0)
chk['predicted_now'] = run(chk.compound.tolist())
chk.to_csv(os.path.join(here, 'byt5skt_recipe_check.csv'), index=False)
print('done')
