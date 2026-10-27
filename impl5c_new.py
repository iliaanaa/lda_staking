"""
impl5c_new.py — ΣΥΓΚΡΙΣΗ ΣΤΡΑΤΗΓΙΚΩΝ ΕΠΙΛΟΓΗΣ ΥΠΟΧΩΡΩΝ (ΜΕ ANOVA).
Top30 / mRMR30 / Diversity30 / Sparse L1 — όλες πάνω στις ΙΔΙΕΣ 100 προβολές ανά split.

Έξοδοι:
  subspace_selection_comparison.csv            -> mean (+std) ανά dataset (ίδιες στήλες με πριν + *_F1_std)
  subspace_selection_comparison_per_split.csv  -> όλα τα runs
"""
import pandas as pd

from common import out_name, overall_summary, run_mode_comparison, save_dataset_checks, wide_summary

USE_ANOVA = True
K_PROJECTIONS = 100
TOP_SELECTED_PROJECTIONS = 30
MODES = {  # όνομα: (στρατηγική επιλογής, L1 meta-learner)
    'top30': ('top_f1', False),
    'mrmr30': ('mrmr', False),
    'diversity30': ('diversity', False),
    'l1_sparse': ('l1_sparse', True),
}
LABELS = {'top30': 'Top30_OOF', 'mrmr30': 'mRMR30', 'diversity30': 'Diversity30', 'l1_sparse': 'Sparse_L1'}

if __name__ == '__main__':
  print('    ADVANCED SUBSPACE SELECTION BENCHMARK (ΜΕ ANOVA, Repeated CV, F1)\n', flush=True)
  df, infos = run_mode_comparison(MODES, USE_ANOVA, K_PROJECTIONS, TOP_SELECTED_PROJECTIONS)

  save_dataset_checks(infos)
  df.to_csv(out_name('subspace_selection_comparison_per_split.csv'), index=False)
  summary = wide_summary(df, LABELS)
  summary.to_csv(out_name('subspace_selection_comparison.csv'), index=False)

  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)
  print('   ΣΥΓΚΡΙΣΗ ΜΕΘΟΔΩΝ ΕΠΙΛΟΓΗΣ ΥΠΟΧΩΡΩΝ (F1 SCORE, μέσος όρος πάνω στα splits)')
  print(summary.to_string(index=False))
  print(overall_summary(df, ['Mode']).to_string(index=False), flush=True)
