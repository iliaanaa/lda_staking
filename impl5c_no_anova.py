"""
impl5c_no_anova.py — LDA STACKING ABLATION STUDY (TOP 30 vs RANDOM 30 vs ALL 100), ΜΕ και ΧΩΡΙΣ ANOVA.

Και τα τρία modes χρησιμοποιούν ΤΙΣ ΙΔΙΕΣ 100 προβολές σε κάθε split (πριν κάθε mode
έπαιρνε διαφορετικές προβολές, άρα η σύγκριση μπερδευόταν με την τύχη).

Έξοδοι:
  lda_stacking_ablation_study_1.csv   -> με ANOVA, mean (+std) ανά dataset
  lda_stacking_no_anova.csv           -> χωρίς ANOVA, mean (+std) ανά dataset
  lda_stacking_ablation_per_split.csv -> όλα τα runs
"""
import pandas as pd

from common import out_name, overall_summary, run_mode_comparison, save_dataset_checks, wide_summary

K_PROJECTIONS = 100
TOP_SELECTED_PROJECTIONS = 30
MODES = {  # όνομα: (στρατηγική επιλογής, L1 meta-learner)
    'top30': ('top_f1', False),
    'random30': ('random', False),
    'all100': ('all', False),
}
LABELS = {'top30': 'Impl5a_Top30', 'random30': 'Impl5b_Rand30', 'all100': 'Impl5c_All100'}

if __name__ == '__main__':
  print('===================================================================')
  print('    LDA STACKING ABLATION: WITH vs WITHOUT ANOVA (Repeated CV, F1)')
  print('===================================================================\n', flush=True)

  df_anova, infos = run_mode_comparison(MODES, True, K_PROJECTIONS, TOP_SELECTED_PROJECTIONS)
  df_noa, _ = run_mode_comparison(MODES, False, K_PROJECTIONS, TOP_SELECTED_PROJECTIONS)

  save_dataset_checks(infos)
  pd.concat([df_anova, df_noa]).to_csv(out_name('lda_stacking_ablation_per_split.csv'), index=False)
  wide_summary(df_anova, LABELS).to_csv(out_name('lda_stacking_ablation_study_1.csv'), index=False)
  wide_summary(df_noa, LABELS).to_csv(out_name('lda_stacking_no_anova.csv'), index=False)

  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)
  print('   [1] WITH ANOVA (lda_stacking_ablation_study_1.csv)')
  print(wide_summary(df_anova, LABELS).to_string(index=False))
  print(overall_summary(df_anova, ['Mode']).to_string(index=False))
  print('\n   [2] WITHOUT ANOVA (lda_stacking_no_anova.csv)')
  print(wide_summary(df_noa, LABELS).to_string(index=False))
  print(overall_summary(df_noa, ['Mode']).to_string(index=False), flush=True)
