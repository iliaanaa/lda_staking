"""
lambda_sensitivity.py — Ευαισθησία της στρατηγικής Diversity στο λ (βλ. PREREGISTRATION_lambda.md).

Για κάθε split: παράγονται μία φορά K_MAX = 500 προβολές (όπως στο grid_search.py) και για κάθε
λ ∈ LAMBDAS και κάθε (K, M) του πλέγματος τρέχει η επιλογή Diversity + meta-learner.
  - λ = 0 υλοποιείται ως 'top_f1' (ακριβώς ίδιο με το Top_F1 του grid_search.py). Έτσι το
    λ = 0 με K = 100, M = 30 πρέπει να δώσει τα ΙΔΙΑ F1 με το grid search (έλεγχος συνέπειας).
  - λ = 0.25 είναι το Diversity του grid search (ίδια αποτελέσματα, δεύτερος έλεγχος συνέπειας).

Εκτέλεση:
  python lambda_sensitivity.py              -> με ANOVA
  python lambda_sensitivity.py --no-anova   -> χωρίς ANOVA

ΣΥΝΕΧΕΙΑ ΜΕΤΑ ΑΠΟ ΔΙΑΚΟΠΗ: κάθε ολοκληρωμένο split γράφεται αμέσως στο CSV. Αν το script
σταματήσει, το ξανατρέχεις και συνεχίζει. Για νέα εκκίνηση σβήσε το CSV.

Έξοδος:
  lambda_sensitivity_all[_no_anova].csv  -> ένα row ανά dataset / split / λ / K / M
Η ανάλυση γίνεται ξεχωριστά με το analyze_lambda.py.
"""
import os
import sys

import pandas as pd

from common import (out_name, DATASET_CONFIGS, generate_projections, iter_splits, load_dataset, meta_stack,
                    preprocess, timer)
from common import select_subspaces

LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
GRID_CONFIGS = [
    {'K': 100, 'M_list': [10, 30, 50]},
    {'K': 200, 'M_list': [10, 30, 50, 100]},
    {'K': 300, 'M_list': [10, 30, 50, 100, 150]},
    {'K': 500, 'M_list': [10, 30, 50, 100, 200]},
]
K_MAX = max(g['K'] for g in GRID_CONFIGS)

if __name__ == '__main__':
  USE_ANOVA = '--no-anova' not in sys.argv
  OUT = out_name(f"lambda_sensitivity_all{'' if USE_ANOVA else '_no_anova'}.csv")
  print(f"    LAMBDA SENSITIVITY ({'ΜΕ' if USE_ANOVA else 'ΧΩΡΙΣ'} ANOVA) — λ = {LAMBDAS}\n", flush=True)

  done = set()
  if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(zip(prev['Dataset'], prev['Repeat'], prev['Fold']))
    print(f'    [RESUME] {len(done)} ολοκληρωμένα splits — συνέχεια.\n', flush=True)

  for cfg in DATASET_CONFIGS:
    d_name = cfg['display_name']
    print(f'>>> [DATASET]: {d_name.upper()}', flush=True)
    elapsed = timer()
    try:
      X, y, num_classes, info = load_dataset(cfg)
      for rep, fold, split_seed, tr, te in iter_splits(y):
        if (d_name, rep, fold) in done:
          continue
        X_tr, X_te = preprocess(X[tr], X[te], y[tr], cfg['k_genes'], use_anova=USE_ANOVA)
        p_tr_all, p_te_all, p_sc_all = generate_projections(
            X_tr, X_te, y[tr], num_classes, cfg['d_prime'], K_MAX, seed=split_seed)
        rows = []
        for lam in LAMBDAS:
          for g in GRID_CONFIGS:
            k = g['K']
            p_tr, p_te, p_sc = p_tr_all[:k], p_te_all[:k], p_sc_all[:k]
            for m in g['M_list']:
              if lam == 0.0:
                idx = select_subspaces('top_f1', p_tr, p_sc, num_classes, m, seed=split_seed)
              else:
                idx = select_subspaces('diversity', p_tr, p_sc, num_classes, m, seed=split_seed, lam=lam)
              res = meta_stack(p_tr, p_te, idx, y[tr], y[te], num_classes)
              rows.append({'Dataset': d_name, 'Repeat': rep, 'Fold': fold, 'Lambda': lam,
                           'K_Projections': k, 'M_Selected': m,
                           'Accuracy': round(res['Accuracy'], 4), 'F1_Score': round(res['F1_Score'], 4),
                           'Train_OOF_F1': round(res['Train_OOF_F1'], 4)})
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        means = pd.DataFrame(rows).groupby('Lambda').F1_Score.mean()
        print(f'    [rep {rep} fold {fold}] μέσο F1 ανά λ: ' +
              ' | '.join(f'{l:g}={v:.3f}' for l, v in means.items()), flush=True)
      print(f'    --> Ολοκλήρωση {d_name} σε {elapsed():.1f}s\n', flush=True)
    except Exception as e:
      print(f'    [ΣΦΑΛΜΑ] {d_name}: {e}\n', flush=True)

  print(f'Αποθηκεύτηκε: {OUT}. Για την ανάλυση: python analyze_lambda.py', flush=True)
