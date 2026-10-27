"""
check_overlap.py — Έλεγχος για ΚΟΙΝΑ ΔΕΙΓΜΑΤΑ ανάμεσα στα datasets (9 αρχικά + 37 νέα).
ΔΕΝ τρέχει κανένα μοντέλο. Συγκρίνει μόνο τα προφίλ έκφρασης των δειγμάτων.

Γιατί δεν αρκεί σύγκριση των GSE: τα περισσότερα αρχικά datasets (Golub, Alon, Singh, Shipp,
Pomeroy, van 't Veer) διανεμήθηκαν εκτός GEO, άρα δεν έχουν GSE για να συγκριθεί. Το ίδιο
δείγμα μπορεί όμως να έχει ανέβει αργότερα σε κάποιο GSE.

Μέθοδος (για κάθε ζεύγος datasets με >= 1000 κοινά probes, π.χ. ίδια πλατφόρμα Affymetrix):
  1. σε κάθε δείγμα, τα γονίδια μετατρέπονται σε τάξεις (ranks) -> ανεξάρτητο από την κανονικοποίηση,
  2. κάθε γονίδιο κεντράρεται με τη διάμεσο ΤΟΥ ΔΙΚΟΥ ΤΟΥ dataset -> αφαιρείται το κοινό προφίλ ιστού,
  3. Pearson ανάμεσα σε κάθε δείγμα του A και κάθε δείγμα του B, στα κοινά probes.
  Μη σχετιζόμενα δείγματα δίνουν χαμηλή συσχέτιση. Το ίδιο δείγμα (ή το ίδιο RNA) δίνει πολύ υψηλή.

Εκτέλεση:   python check_overlap.py
Έξοδοι:     overlap_pairs.csv    -> ένα row ανά ζεύγος datasets: κοινά probes, μέγιστη και 99η εκατοστιαία r
            overlap_suspects.csv -> ζεύγη δειγμάτων με r >= SUSPECT_R (υποψήφια διπλότυπα)
"""
import os
import re

import numpy as np
import pandas as pd

from common import CUMIDA_DIR, NEW_OPENML_CONFIGS, CUMIDA_CONFIGS, ORIGINAL_CONFIGS, fetch_microarray_robust

MIN_SHARED = 1000
SUSPECT_R = 0.80


def norm_name(c):
  """Ομοιόμορφα ονόματα probes: 'X1007_s_at' (R make.names) -> '1007_s_at'."""
  c = str(c).strip().strip('\'"')
  c = re.sub(r'^X(?=\d)', '', c)
  return c.replace('.', '-').lower()


def load_named(cfg):
  if 'cumida_file' in cfg:
    path = os.path.join(CUMIDA_DIR, cfg['cumida_file'])
    df = pd.read_csv(path)
    ids = df['samples'].astype(str).values if 'samples' in df.columns else np.arange(len(df)).astype(str)
    df = df.drop(columns=[c for c in ('samples', 'type') if c in df.columns])
  else:
    data, _ = fetch_microarray_robust(cfg['identifiers'])
    df = data.data.copy()
    ids = np.array([f'row{i}' for i in range(len(df))])
  df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
  df.columns = [norm_name(c) for c in df.columns]
  df = df.loc[:, ~pd.Index(df.columns).duplicated()]
  return df.astype(np.float32), ids


def standardize(df):
  """rank ανά δείγμα -> κεντράρισμα ανά γονίδιο (διάμεσος dataset) -> μοναδιαία νόρμα ανά δείγμα."""
  r = df.rank(axis=1).fillna(df.shape[1] / 2).values.astype(np.float32)
  r -= np.median(r, axis=0, keepdims=True)
  r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
  return r


if __name__ == '__main__':
  configs = ORIGINAL_CONFIGS + NEW_OPENML_CONFIGS + CUMIDA_CONFIGS
  data = {}
  for cfg in configs:
    name = cfg['display_name']
    try:
      df, ids = load_named(cfg)
      data[name] = (df, ids)
      print(f'  φορτώθηκε {name:<35} {df.shape}', flush=True)
    except Exception as e:
      print(f'  [ΣΦΑΛΜΑ] {name}: {e}', flush=True)

  names = list(data)
  pairs, suspects = [], []
  for i in range(len(names)):
    for j in range(i + 1, len(names)):
      a, b = names[i], names[j]
      dfa, ida = data[a]
      dfb, idb = data[b]
      shared = dfa.columns.intersection(dfb.columns)
      if len(shared) < MIN_SHARED:
        continue
      za, zb = standardize(dfa[shared]), standardize(dfb[shared])
      r = za @ zb.T
      pairs.append({'Dataset_A': a, 'Dataset_B': b, 'Shared_probes': len(shared),
                    'Max_r': round(float(r.max()), 3), 'P99_r': round(float(np.percentile(r, 99)), 3),
                    'Median_r': round(float(np.median(r)), 3), 'N_pairs_r>=thr': int((r >= SUSPECT_R).sum())})
      for ia, ib in zip(*np.where(r >= SUSPECT_R)):
        suspects.append({'Dataset_A': a, 'Sample_A': ida[ia], 'Dataset_B': b, 'Sample_B': idb[ib],
                         'r': round(float(r[ia, ib]), 3)})
      print(f'  {a:<32} × {b:<32} κοινά={len(shared):>6}  max r={r.max():.3f}  '
            f'ύποπτα={int((r >= SUSPECT_R).sum())}', flush=True)

  pd.DataFrame(pairs).to_csv('overlap_pairs.csv', index=False)
  pd.DataFrame(suspects, columns=['Dataset_A', 'Sample_A', 'Dataset_B', 'Sample_B', 'r']).to_csv(
      'overlap_suspects.csv', index=False)
  no_names = [n for n in names if not any(n in (p['Dataset_A'], p['Dataset_B']) for p in pairs)]
  print(f'\nΖεύγη που ελέγχθηκαν: {len(pairs)}.  Ύποπτα ζεύγη δειγμάτων (r >= {SUSPECT_R}): {len(suspects)}')
  if no_names:
    print('Δεν βρέθηκαν κοινά ονόματα probes με κανένα άλλο dataset (δεν ελέγχθηκαν): ' + ', '.join(no_names))
  print('Αποθηκεύτηκαν: overlap_pairs.csv, overlap_suspects.csv')
