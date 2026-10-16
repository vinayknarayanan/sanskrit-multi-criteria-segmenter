#!/usr/bin/env python3
"""Fine-tune ByT5 on the Frontiers training data, then predict the 3,259-item evaluation sample.

Purpose (Thesis A, controlled comparison): same architecture as ByT5-Sanskrit, but trained on the Frontiers
convention instead of DCS. Prediction: under strict Frontiers scoring this model beats DCS-trained ByT5-Sanskrit;
under convention-aware scoring the gap shrinks or reverses. Output is written in the same format as
results_byt5.csv, so rescore.py can score it unchanged.

Defaults are set for a 4-6 GB laptop GPU (e.g. RTX 3050 Laptop): Adafactor (factored, low-memory optimizer, the
standard for T5), gradient checkpointing, batch 4 x 8 accumulation (effective 32), bf16, one model-only checkpoint.
On Colab/bigger GPUs you can raise --bs and lower --grad-accum. Expect ~4-8 h for 3 epochs on a laptop 3050.

Usage
  pip install -q transformers==4.44.2 datasets accelerate sentencepiece pandas
  python finetune_byt5_frontiers.py --data dataset.csv --eval shared_eval_sample.csv --out runs/byt5_frontiers
  # arm 2 (optional): start from the DCS-trained checkpoint instead of plain ByT5
  python finetune_byt5_frontiers.py ... --init <path-or-hub-id-of-ByT5-Sanskrit> --out runs/byt5skt_frontiers
  # predict only, from a finished run
  python finetune_byt5_frontiers.py --data dataset.csv --eval shared_eval_sample.csv --out runs/byt5_frontiers --predict-only

Design decisions (all printed to runs/.../config.json):
  * Evaluation items are removed from training/dev by surface form AND by gold split, so the model never sees them.
  * Target = gold split with '+' between pieces, exactly as in the gold. Prediction is converted to space-separated
    pieces for the results file (the format results_byt5.csv uses).
  * IAST in, IAST out (ByT5 is byte-level; diacritics are multi-byte, so max lengths are in bytes).
  * bf16 if the GPU supports it (3060 does), else fp32. fp16 is avoided: ByT5 is unstable in fp16.
  * Seeded; dev split 5% of the remaining data for early stopping on loss.
"""
import argparse, json, os, random, sys
os.environ.setdefault('USE_TF', '0')            # never let an installed TensorFlow/Keras break transformers
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
import numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback)

COMP_COLS = ['compound', 'word', 'surface', 'input', 'text', 'sandhied']
SPLIT_COLS = ['gold_split', 'split', 'segmented', 'target', 'output', 'gold']

def pick(cols, cands, what):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    sys.exit(f'cannot find the {what} column in {list(cols)}; rename it to one of {cands}')

def norm(s): return ' '.join(str(s).strip().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='full Frontiers dataset CSV (compound, gold_split)')
    ap.add_argument('--eval', required=True, help='shared_eval_sample.csv (the 3,259 items to hold out and predict)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default='google/byt5-small')
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--optim', default='adafactor', help='adafactor (low memory) or adamw_torch')
    ap.add_argument('--no-grad-ckpt', action='store_true')
    ap.add_argument('--max-src', type=int, default=256, help='bytes')
    ap.add_argument('--max-tgt', type=int, default=288, help='bytes')
    ap.add_argument('--beams', type=int, default=4)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--limit', type=int, default=0, help='debug: train on N rows only')
    ap.add_argument('--predict-only', action='store_true')
    ap.add_argument('--predict-tags', default='', help='comma-separated convention tags for a tagged model, e.g. fr,dcs -> inputs prefixed "<fr> ", "<dcs> "; one results file per tag')
    ap.add_argument('--resume', action='store_true', help='continue from the last checkpoint in <out>/ckpt (same settings)')
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------------- data, with the evaluation items removed
    df = pd.read_csv(a.data); cc, sc = pick(df.columns, COMP_COLS, 'surface'), pick(df.columns, SPLIT_COLS, 'gold split')
    df = pd.DataFrame({'src': df[cc].map(norm), 'tgt': df[sc].map(norm)}).dropna()
    df = df[(df.src != '') & (df.tgt.str.contains(r'\+', regex=True))].drop_duplicates()
    ev = pd.read_csv(a.eval)
    ev_src, ev_tgt = set(ev.compound.map(norm)), set(ev.gold_split.map(norm))
    before = len(df)
    df = df[~df.src.isin(ev_src) & ~df.tgt.isin(ev_tgt)]
    removed = before - len(df)
    df = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    if a.limit: df = df.head(a.limit)
    n_dev = max(200, int(len(df) * 0.05))
    dev, train = df.iloc[:n_dev], df.iloc[n_dev:]
    cfg = dict(vars(a), n_total_after_dedup=before, removed_eval_overlap=removed, n_train=len(train), n_dev=len(dev),
               n_eval=len(ev), torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    tok = AutoTokenizer.from_pretrained(a.init)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def enc(b):
        x = tok(b['src'], max_length=a.max_src, truncation=True)
        x['labels'] = tok(text_target=b['tgt'], max_length=a.max_tgt, truncation=True)['input_ids']
        return x

    model_dir = os.path.join(a.out, 'model')
    if not a.predict_only:
        model = AutoModelForSeq2SeqLM.from_pretrained(a.init)
        for p in model.parameters():          # ByT5 ships some transposed (non-contiguous) weights; safetensors refuses them
            p.data = p.data.contiguous()
        tr = Dataset.from_pandas(train).map(enc, batched=True, remove_columns=['src', 'tgt'])
        dv = Dataset.from_pandas(dev).map(enc, batched=True, remove_columns=['src', 'tgt'])
        args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(a.out, 'ckpt'), per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
            gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs,
            lr_scheduler_type='linear', warmup_ratio=0.03, weight_decay=0.0, bf16=bf16, optim=a.optim,
            gradient_checkpointing=not a.no_grad_ckpt, save_only_model=False, save_safetensors=False,   # full state -> resumable after a crash/reboot
            eval_strategy='steps', eval_steps=1000, save_steps=1000, save_total_limit=1, logging_steps=100,
            load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
            predict_with_generate=False, seed=a.seed, report_to='none', dataloader_num_workers=0 if os.name == 'nt' else 2)
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tr, eval_dataset=dv, tokenizer=tok,
                                 data_collator=DataCollatorForSeq2Seq(tok, model=model),
                                 callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        trainer.train(resume_from_checkpoint=True if a.resume else None)
        trainer.save_model(model_dir); tok.save_pretrained(model_dir)

    # ---------------------------------------------------------------- predict the evaluation sample
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir if os.path.isdir(model_dir) else a.init)
    dev_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev_).eval()
    if bf16: model = model.to(torch.bfloat16)
    for tag in (a.predict_tags.split(',') if a.predict_tags else ['plain']):
        prefix = '' if tag == 'plain' else f'<{tag}> '
        preds = []
        srcs = [prefix + x for x in ev.compound.map(norm)]
        for i in range(0, len(srcs), 16):
            batch = tok(srcs[i:i + 16], return_tensors='pt', padding=True, truncation=True, max_length=a.max_src).to(dev_)
            with torch.no_grad():
                out = model.generate(**batch, max_length=a.max_tgt, num_beams=a.beams)
            preds += tok.batch_decode(out, skip_special_tokens=True)
            print(f'predicted {min(i + 16, len(srcs))}/{len(srcs)}', end='\r')
        res = ev.copy()
        res['predicted'] = [' '.join(p for p in s.split('+') if p.strip()) for s in preds]   # space-separated, like results_byt5.csv
        res['pred_piece_count'] = res.predicted.str.split().str.len()
        res['acc_exact'] = [norm(p) == norm(g) for p, g in zip(preds, res.gold_split)]
        path = os.path.join(a.out, 'results_byt5_frontiers.csv' if tag == 'plain' else f'results_{tag}.csv'); res.to_csv(path, index=False)
        print(f'\nwrote {path}; exact match {res.acc_exact.mean()*100:.1f}%')

if __name__ == '__main__':
    main()
