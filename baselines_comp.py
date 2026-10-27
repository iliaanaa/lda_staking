"""
baselines_comp.py — Κλασικά baseline μοντέλα (ίδια splits & ίδια μετρική F1 με το LDA Stacking).

Έξοδοι:
  baselines_comparison_per_split.csv  -> ένα row ανά dataset / split / σενάριο / μοντέλο
  baselines_comparison_results.csv    -> mean & std (πάνω στα splits) ανά dataset / σενάριο / μοντέλο
  datasets_check.csv                  -> τι φορτώθηκε πραγματικά από το OpenML
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC, SVC

from common import (out_name, DATASET_CONFIGS, SEED, calc_f1, iter_splits, load_dataset, overall_summary,
                    per_dataset_summary, preprocess, save_dataset_checks, timer, tune_threshold)


def get_baseline_models(seed=SEED):
  return {
      'SVM_Linear': LinearSVC(C=1.0, dual='auto', random_state=seed),
      'SVM_RBF': SVC(kernel='rbf', C=1.0, random_state=seed),
      'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
      'Hist_Gradient_Boosting': HistGradientBoostingClassifier(random_state=seed),
      'LASSO_Logistic': LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', random_state=seed, max_iter=500),
  }


def evaluate_baseline_model(model, X_tr, X_te, y_train, y_test, num_classes):
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
  has_proba = hasattr(model, 'predict_proba')
  method = 'predict_proba' if has_proba else 'decision_function'
  oof = cross_val_predict(model, X_tr, y_train, cv=cv, method=method)

  best_thresh = 0.5 if has_proba else 0.0
  if num_classes == 2:
    scores_val = oof[:, 1] if has_proba else oof
    thresh_range = np.linspace(0.1, 0.9, 81) if has_proba else np.linspace(scores_val.min(), scores_val.max(), 81)
    best_thresh, _ = tune_threshold(scores_val, y_train, num_classes, thresh_range)

  model.fit(X_tr, y_train)
  test_scores = model.predict_proba(X_te) if has_proba else model.decision_function(X_te)
  if num_classes == 2:
    s = test_scores[:, 1] if has_proba else test_scores
    y_pred = (s >= best_thresh).astype(int)
  else:
    y_pred = np.argmax(test_scores, axis=1)
  return accuracy_score(y_test, y_pred), calc_f1(y_test, y_pred, num_classes)


if __name__ == '__main__':
  print('         BENCHMARKING CLASSICAL BASELINE MODELS (Repeated Stratified CV, F1 score)\n', flush=True)
  results, infos = [], []

  for config in DATASET_CONFIGS:
    d_name, k_genes = config['display_name'], config['k_genes']
    print(f'>>> [BENCHMARKING DATASET]: {d_name.upper()}', flush=True)
    elapsed = timer()
    try:
      X, y, num_classes, info = load_dataset(config)
      infos.append(info)
      for rep, fold, _, tr, te in iter_splits(y):
        for use_anova in [True, False]:
          scenario = 'ANOVA' if use_anova else 'RawSpace'
          X_tr, X_te = preprocess(X[tr], X[te], y[tr], k_genes, use_anova)
          for m_name, model in get_baseline_models().items():
            acc, f1 = evaluate_baseline_model(model, X_tr, X_te, y[tr], y[te], num_classes)
            results.append({'Dataset': d_name, 'Repeat': rep, 'Fold': fold, 'Scenario': scenario,
                            'Model': m_name, 'Accuracy': round(acc, 4), 'F1_Score': round(f1, 4)})
        last = [r for r in results if r['Dataset'] == d_name and r['Repeat'] == rep and r['Fold'] == fold]
        print(f'    [rep {rep} fold {fold}] ' + ' | '.join(f"{r['Scenario'][:3]}-{r['Model'][:6]}={r['F1_Score']:.3f}" for r in last), flush=True)
      print(f'    --> Ολοκλήρωση {d_name} σε {elapsed():.1f}s\n', flush=True)
    except Exception as e:
      print(f'    [ΣΦΑΛΜΑ] {d_name}: {e}\n', flush=True)

  save_dataset_checks(infos)
  df = pd.DataFrame(results)
  df.to_csv(out_name('baselines_comparison_per_split.csv'), index=False)
  per_dataset_summary(df, ['Scenario', 'Model']).to_csv(out_name('baselines_comparison_results.csv'), index=False)

  print('                   ΣΥΓΚΕΝΤΡΩΤΙΚΑ BASELINE RESULTS (F1 score)')
  print(overall_summary(df, ['Scenario', 'Model']).to_string(index=False), flush=True)
