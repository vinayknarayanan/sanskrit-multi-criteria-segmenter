"""Audit of the released Frontiers file (Sections 3 and 5.5): row count, written forms with two divisions,
duplicates and non-IAST rows. Usage: python tools/frontiers_audit.py data/third_party/frontiers_dataset.csv"""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'data/third_party/frontiers_dataset.csv')
amb = d.drop_duplicates().groupby('compound').split.nunique(); amb = set(amb[amb > 1].index)
print('rows', len(d), '| distinct forms', d.compound.nunique())
print('forms with two or more divisions', len(amb), '| rows of them', d.compound.isin(amb).sum(),
      '| rows left after removing them', (~d.compound.isin(amb)).sum())
print('exact duplicate rows', d.duplicated().sum())
print('rows with characters outside IAST', d.compound.str.contains(r"[^a-zāīūṛṝḷḹṅñṭḍṇśṣṃḥ'’]").sum())
print('reported in the article: 89,656')
