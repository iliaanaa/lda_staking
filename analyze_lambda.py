"""
analyze_lambda.py — Η ΠΡΟΚΑΘΟΡΙΣΜΕΝΗ ανάλυση του PREREGISTRATION_lambda.md.
Διαβάζει lambda_sensitivity_all.csv και/ή lambda_sensitivity_all_no_anova.csv (όσα υπάρχουν).
Απαιτεί: pip install baycomp

Έξοδος: lambda_analysis_report.txt (και στην οθόνη).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROPE = 0.01
KEYS = ['Dataset', 'Repeat', 'Fold']

try:
  from baycomp import SignedRankTest
except ImportError:
  SignedRankTest = None


def bayes(x, y):
  """P(x καλύτερο), P(πρακτικά ίσο), P(y καλύτερο) — Bayesian signed-rank, ROPE = 0.01."""
  if SignedRankTest is None:
    return None
  # baycomp.SignedRankTest.probs(a, b) επιστρέφει (P(a > b), P(rope), P(b > a)) — επαληθευμένο
  p_x_better, p_rope, p_y_better = SignedRankTest.probs(np.asarray(x), np.asarray(y), rope=ROPE)
  return p_x_better, p_rope, p_y_better


def wil(x, y):
  d = np.asarray(x) - np.asarray(y)
  return np.nan if np.all(d == 0) else wilcoxon(x, y).pvalue


def select_on_train(df, by):
  """Μέγιστο Train_OOF_F1. Ισοπαλία: μικρότερο λ, μετά μικρότερο K, μετά μικρότερο M (§3)."""
  df = df.sort_values(['Train_OOF_F1', 'Lambda', 'K_Projections', 'M_Selected'],
                      ascending=[False, True, True, True])
  return df.groupby(by, sort=False).head(1)


def decision(pb):
  if pb is None:
    return 'baycomp δεν είναι εγκατεστημένο'
  if pb[0] >= 0.95:
    return 'Η H1 ΥΠΟΣΤΗΡΙΖΕΤΑΙ'
  if pb[1] >= 0.95:
    return 'Η H1 ΑΠΟΡΡΙΠΤΕΤΑΙ (πρακτικά ισοδύναμα)'
  return 'ΑΔΙΕΥΚΡΙΝΙΣΤΟ (ανεπαρκή δεδομένα)'


def analyze(path, title, out):
  df = pd.read_csv(path)
  out.append(f"\n{'=' * 78}\n{title}   ({path}, {len(df)} runs, {df.Dataset.nunique()} datasets)\n{'=' * 78}")

  # ---- ΚΥΡΙΑ (H1): λ ρυθμισμένο στο train vs λ = 0, με K, M ρυθμισμένα στο train ----
  tuned = select_on_train(df, KEYS).groupby('Dataset').F1_Score.mean()
  lam0 = select_on_train(df[df.Lambda == 0], KEYS).groupby('Dataset').F1_Score.mean()
  pb = bayes(tuned, lam0)
  out.append('\n[ΚΥΡΙΑ — H1] λ ρυθμισμένο στο train  vs  λ = 0 (Top_F1), K, M ρυθμισμένα στο train')
  out.append(f'  Μέσο F1: λ-tuned = {tuned.mean():.4f}   λ=0 = {lam0.mean():.4f}   διαφορά = {(tuned - lam0).mean():+.4f}')
  out.append(f'  Νίκες / ισοπαλίες / ήττες (datasets): {(tuned > lam0 + 1e-9).sum()} / '
             f'{(abs(tuned - lam0) <= 1e-9).sum()} / {(tuned < lam0 - 1e-9).sum()}')
  if pb:
    out.append(f'  Bayesian signed-rank (ROPE ±{ROPE}): P(καλύτερο) = {pb[0]:.3f}, '
               f'P(πρακτικά ίσο) = {pb[1]:.3f}, P(χειρότερο) = {pb[2]:.3f}')
  out.append(f'  Wilcoxon p = {wil(tuned, lam0):.4f}')
  out.append(f'  ΑΠΟΦΑΣΗ: {decision(pb)}')
  chosen = select_on_train(df, KEYS).Lambda.value_counts().sort_index()
  out.append('  Πόσο συχνά επιλέχθηκε κάθε λ στο train: ' + ', '.join(f'λ={l:g}: {n}' for l, n in chosen.items()))

  # ---- ΔΕΥΤΕΡΕΥΟΥΣΑ (H2): ανά σταθερό λ, K, M ρυθμισμένα στο train ----
  out.append('\n[ΔΕΥΤΕΡΕΥΟΥΣΑ — H2] Ανά σταθερό λ (K, M ρυθμισμένα στο train)')
  per_lam = select_on_train(df, KEYS + ['Lambda']).groupby(['Dataset', 'Lambda']).F1_Score.mean().unstack()
  ranks = per_lam.rank(axis=1, ascending=False).mean()
  for l in per_lam.columns:
    out.append(f'  λ = {l:<5g}  μέσο F1 = {per_lam[l].mean():.4f}   μέση κατάταξη = {ranks[l]:.2f}')
  out.append(f'  Friedman ({per_lam.shape[1]} τιμές λ): p = {friedmanchisquare(*[per_lam[c] for c in per_lam]).pvalue:.4f}')

  # ---- ΔΕΥΤΕΡΕΥΟΥΣΑ (H3): M = 10, [λ = 0.25] − [λ = 0] για K = 100 και K = 500 ----
  out.append('\n[ΔΕΥΤΕΡΕΥΟΥΣΑ — H3] M = 10: [λ = 0.25] − [λ = 0]')
  fixed = df.groupby(['Dataset', 'Lambda', 'K_Projections', 'M_Selected']).F1_Score.mean()
  ks = sorted(df.K_Projections.unique())
  for K in ([100, 500] if {100, 500} <= set(ks) else [ks[0], ks[-1]]):
    a = fixed.xs((0.25, K, 10), level=['Lambda', 'K_Projections', 'M_Selected'])
    b = fixed.xs((0.0, K, 10), level=['Lambda', 'K_Projections', 'M_Selected'])
    pb3 = bayes(a, b)
    s = f'  K = {K}: διαφορά = {(a - b).mean():+.4f}, νίκες/ήττες = {(a > b).sum()}/{(a < b).sum()}'
    if pb3:
      s += f', P(καλύτερο) = {pb3[0]:.3f}, P(ίσο) = {pb3[1]:.3f}, P(χειρότερο) = {pb3[2]:.3f}'
    out.append(s)

  # ---- Έλεγχοι συνέπειας με τα προηγούμενα grid search ----
  out.append('\n[ΕΛΕΓΧΟΣ ΣΥΝΕΠΕΙΑΣ] λ = 0 ≡ Top_F1 και λ = 0.25 ≡ Diversity του grid_search.py')
  out.append('  (σύγκρινε με subspace_grid_search_all_strategies*.csv — πρέπει να είναι ΙΔΙΑ)')
  for lam, strat in [(0.0, 'top_f1'), (0.25, 'diversity')]:
    v = df[(df.Lambda == lam) & (df.K_Projections == 100) & (df.M_Selected == 30)].F1_Score.mean()
    if not np.isnan(v):
      out.append(f'  λ = {lam:g}, K = 100, M = 30: μέσο F1 πάνω σε όλα τα splits = {v:.4f}')


if __name__ == '__main__':
  if SignedRankTest is None:
    print('ΠΡΟΣΟΧΗ: το baycomp δεν είναι εγκατεστημένο (pip install baycomp). '
          'Θα τρέξουν μόνο οι μη-Bayesian έλεγχοι.')
  out = ['ΠΡΟΚΑΘΟΡΙΣΜΕΝΗ ΑΝΑΛΥΣΗ — PREREGISTRATION_lambda.md']
  from common import out_name   # --datasets new -> διαβάζει τα *_new.csv
  for path, title in [(out_name('lambda_sensitivity_all.csv'), 'ΜΕ ANOVA'),
                      (out_name('lambda_sensitivity_all_no_anova.csv'), 'ΧΩΡΙΣ ANOVA')]:
    if os.path.exists(path):
      analyze(path, title, out)
  text = '\n'.join(out)
  print(text)
  with open(out_name('lambda_analysis_report.txt'), 'w', encoding='utf-8') as f:
    f.write(text)
