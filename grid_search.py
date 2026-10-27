"""
grid_search.py — Grid search K (προβολές) x M (επιλεγμένοι υποχώροι) για όλες τις στρατηγικές.

Διόρθωση 4: η επιλογή του (K, M) γίνεται ΜΕΣΑ στο training set.
  Για κάθε split και στρατηγική επιλέγεται ο συνδυασμός με το υψηλότερο Train_OOF_F1
  (OOF F1 του meta-learner στο train fold). Το F1 στο test fold του επιλεγμένου
  συνδυασμού είναι η αμερόληπτη εκτίμηση. Το test set δεν χρησιμοποιείται για επιλογή.

Εκτέλεση:
  python grid_search.py              -> με ANOVA (όπως πριν)
  python grid_search.py --no-anova   -> χωρίς ANOVA (raw χώρος, όλα τα γονίδια)

Έξοδοι (με --no-anova τα ονόματα παίρνουν την κατάληξη _no_anova, ώστε να ΜΗΝ
αντικαθίστανται τα αποτελέσματα με ANOVA):
  subspace_grid_search_all_strategies[_no_anova].csv  -> όλα τα runs (ανά split), με Train_OOF_F1
  subspace_grid_search_summary[_no_anova].csv         -> mean & std (πάνω στα splits) ανά dataset/στρατηγική/K/M
  subspace_grid_search_selected[_no_anova].csv        -> ο (K, M) που επιλέχθηκε στο train ανά split + test F1
"""
import sys

import pandas as pd

from common import (out_name, DATASET_CONFIGS, generate_projections, iter_splits, load_dataset, meta_stack,
                    overall_summary, per_dataset_summary, preprocess, save_dataset_checks,
                    select_subspaces, timer)

GRID_CONFIGS = [
    {'K': 100, 'M_list': [10, 30, 50]},
    {'K': 200, 'M_list': [10, 30, 50, 100]},
    {'K': 300, 'M_list': [10, 30, 50, 100, 150]},
    {'K': 500, 'M_list': [10, 30, 50, 100, 200]},
]
K_MAX = max(g['K'] for g in GRID_CONFIGS)
STRATEGIES = ['diversity', 'top_f1', 'mrmr', 'l1_sparse']


def select_config_on_train(df):
  """Ανά dataset/split/στρατηγική: max Train_OOF_F1. Ισοπαλία -> μικρότερο K, μετά μικρότερο M."""
  df = df.sort_values(['Train_OOF_F1', 'K_Projections', 'M_Selected'], ascending=[False, True, True])
  return df.groupby(['Dataset', 'Repeat', 'Fold', 'Strategy'], sort=False).head(1).sort_values(
      ['Dataset', 'Strategy', 'Repeat', 'Fold'])


if __name__ == '__main__':
  USE_ANOVA = '--no-anova' not in sys.argv
  SUFFIX = '' if USE_ANOVA else '_no_anova'
  print('===================================================================')
  print(f"    GRID SEARCH: ALL SUBSPACE SELECTION STRATEGIES ({'ΜΕ' if USE_ANOVA else 'ΧΩΡΙΣ'} ANOVA, Repeated CV, F1)")
  print('===================================================================\n', flush=True)
  all_results, infos = [], []

  for cfg in DATASET_CONFIGS:
    d_name = cfg['display_name']
    print(f'>>> [BENCHMARKING DATASET]: {d_name.upper()}', flush=True)
    elapsed = timer()
    try:
      X, y, num_classes, info = load_dataset(cfg)
      infos.append(info)
      for rep, fold, split_seed, tr, te in iter_splits(y):
        X_tr, X_te = preprocess(X[tr], X[te], y[tr], cfg['k_genes'], use_anova=USE_ANOVA)
        t_gen = timer()
        p_tr_all, p_te_all, p_sc_all = generate_projections(
            X_tr, X_te, y[tr], num_classes, cfg['d_prime'], K_MAX, seed=split_seed)
        gen_seconds = t_gen()
        print(f'  [rep {rep} fold {fold}] {K_MAX} προβολές LDA σε {gen_seconds:.1f}s', flush=True)

        for strat in STRATEGIES:
          for g in GRID_CONFIGS:
            k = g['K']
            p_tr, p_te, p_sc = p_tr_all[:k], p_te_all[:k], p_sc_all[:k]
            # l1_sparse χρησιμοποιεί ΟΛΕΣ τις K προβολές -> ένα run ανά K (M = K)
            m_values = [k] if strat == 'l1_sparse' else g['M_list']
            for m in m_values:
              t0 = timer()
              idx = select_subspaces(strat, p_tr, p_sc, num_classes, m, seed=split_seed)
              res = meta_stack(p_tr, p_te, idx, y[tr], y[te], num_classes, l1=(strat == 'l1_sparse'))
              all_results.append({
                  'Dataset': d_name, 'Repeat': rep, 'Fold': fold, 'Strategy': strat,
                  'K_Projections': k, 'M_Selected': m,
                  'Accuracy': round(res['Accuracy'], 4), 'F1_Score': round(res['F1_Score'], 4),
                  'Train_OOF_F1': round(res['Train_OOF_F1'], 4), 'Seconds': round(t0(), 2),
                  # χρόνος παραγωγής των k πρώτων προβολών (αναλογικά) + επιλογή/meta = συνολικό κόστος ρύθμισης
                  'Proj_Gen_Seconds': round(gen_seconds * k / K_MAX, 2),
                  'Total_Seconds': round(gen_seconds * k / K_MAX + t0(), 2),
              })
          best = max((r for r in all_results if r['Dataset'] == d_name and r['Repeat'] == rep
                      and r['Fold'] == fold and r['Strategy'] == strat), key=lambda r: r['Train_OOF_F1'])
          print(f"      [{strat.upper():10s}] επιλογή στο train: K={best['K_Projections']}, M={best['M_Selected']} "
                f"-> test F1 = {best['F1_Score']:.4f}", flush=True)
      print(f'    --> Ολοκλήρωση {d_name} σε {elapsed():.1f}s\n', flush=True)
    except Exception as e:
      print(f'    [ΣΦΑΛΜΑ] {d_name}: {e}\n', flush=True)

  save_dataset_checks(infos, f'datasets_check{SUFFIX}.csv')
  df = pd.DataFrame(all_results)
  df.to_csv(out_name(f'subspace_grid_search_all_strategies{SUFFIX}.csv'), index=False)
  per_dataset_summary(df, ['Strategy', 'K_Projections', 'M_Selected'],
                      ('Accuracy', 'F1_Score', 'Train_OOF_F1', 'Seconds', 'Total_Seconds')).to_csv(
      out_name(f'subspace_grid_search_summary{SUFFIX}.csv'), index=False)
  selected = select_config_on_train(df)
  selected.to_csv(out_name(f'subspace_grid_search_selected{SUFFIX}.csv'), index=False)

  pd.set_option('display.width', 1000)
  print('      ΣΥΓΚΕΝΤΡΩΤΙΚΑ GRID SEARCH (F1 στο test, ανά σταθερό K, M)')
  print(overall_summary(df, ['Strategy', 'K_Projections', 'M_Selected']).to_string(index=False))
  print('\n      ΑΜΕΡΟΛΗΠΤΗ ΕΚΤΙΜΗΣΗ: (K, M) επιλεγμένα στο train, F1 στο test')
  print(overall_summary(selected, ['Strategy']).to_string(index=False))
  print('\n      Πόσο συχνά επιλέχθηκε κάθε (K, M):')
  print(selected.groupby(['Strategy', 'K_Projections', 'M_Selected']).size().rename('Times_Selected')
        .reset_index().to_string(index=False), flush=True)
