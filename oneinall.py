"""
oneinall.py — Νευρωνικές μέθοδοι Impl 1–4 (PANN-EC και Multi-NN), με το ΙΔΙΟ πρωτόκολλο
αξιολόγησης με όλα τα υπόλοιπα πειράματα (common.py):
  - ίδια 9 datasets (σωστά OpenML IDs, έλεγχος N/p)
  - ίδια 10 splits (Repeated Stratified 5-Fold x 2)
  - ίδια μετρική: binary F1 με θετική κλάση την ασθένεια
  - τυχαίες προβολές με τοπικό seed ανά split

Η λογική κάθε μεθόδου είναι ΑΥΤΟΥΣΙΑ από το αρχικό oneinall.py. Αλλάζουν μόνο:
  1. Η μετρική: calc_f1 αντί για macro F1.
  2. Οι προβολές: rng = np.random.default_rng(split_seed) αντί για np.random.normal.
     Τα Impl 1 και 3 (raw) βλέπουν τις ίδιες προβολές μεταξύ τους, και το ίδιο τα Impl 2 και 4 (ANOVA).
  3. Το seed του TensorFlow ορίζεται ανά split, για αναπαραγωγιμότητα.
Το Impl 5 (LDA Stacking) ΔΕΝ περιλαμβάνεται: τα αποτελέσματά του υπάρχουν ήδη στα ίδια splits
(lda_stacking_ablation_per_split.csv, Top30).

ΣΥΝΕΧΕΙΑ ΜΕΤΑ ΑΠΟ ΔΙΑΚΟΠΗ: κάθε split που ολοκληρώνεται γράφεται αμέσως στο
oneinall_per_split.csv. Αν το script σταματήσει (Ctrl+C, κλείσιμο υπολογιστή), ξανατρέξ' το
και θα συνεχίσει από εκεί που έμεινε. Για να ξεκινήσει από την αρχή, σβήσε το CSV.

Έξοδοι:
  oneinall_per_split.csv  -> ένα row ανά dataset / split / μέθοδο
  oneinall_results.csv    -> mean & std (πάνω στα splits) ανά dataset / μέθοδο
"""
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from common import (out_name, DATASET_CONFIGS, SEED, calc_f1, iter_splits, load_dataset, overall_summary,
                    per_dataset_summary, preprocess, save_dataset_checks, timer)

tf.get_logger().setLevel('ERROR')

K_PROJECTIONS = 100
OUT_PER_SPLIT = out_name('oneinall_per_split.csv')

# d'' (διάσταση PCA) για τα Impl 2 και 4, όπως στο αρχικό oneinall.py.
# Τα CNS και Breast δεν υπήρχαν στο αρχικό αρχείο -> 30, η προεπιλεγμένη τιμή PCA της
# μεθόδου PANN-EC (ίδια με τα Impl 1 και 3). Δεν επιλέχθηκε με βάση τα αποτελέσματα.
D_DOUBLE_PRIME = {
    'leukemia': 30, 'colon': 10, 'prostate': 10, 'SMK-CAN-187': 10, 'GLI-85': 20,
    'dlbcl': 10, 'ovarian': 20, 'CNS': 30, 'Breast': 30,
}


def build_mlp_model(input_dim, num_classes):
  model = Sequential([
      Input(shape=(input_dim,)),
      Dense(100, activation='relu'),
      Dropout(0.2),
      Dense(64, activation='relu'),
      Dropout(0.2),
      Dense(num_classes, activation='softmax'),
  ])
  model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model


class _FastTrainer:
  """Ίδια εκπαίδευση με model.fit(epochs=30, batch_size=len(X)), περίπου 35x ταχύτερα.

  Με batch = όλο το train set, το model.fit κάνει ακριβώς ένα βήμα RMSprop ανά epoch.
  Εδώ το ίδιο βήμα γίνεται με compiled tf.function. Το ΙΔΙΟ μοντέλο ξαναχρησιμοποιείται,
  με νέα τυχαία αρχικοποίηση (glorot_uniform / zeros, όπως ένα καινούριο μοντέλο) και
  μηδενισμένο optimizer για κάθε προβολή. Επαληθεύτηκε ότι, με την ίδια αρχικοποίηση,
  δίνει τα ίδια βάρη με το model.fit (μέγιστη διαφορά ~1e-7, στρογγυλοποίηση float32).
  """
  _cache = {}

  def __init__(self, input_dim, num_classes):
    self.model = build_mlp_model(input_dim, num_classes)
    self.opt = self.model.optimizer
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model, opt, loss_fn = self.model, self.opt, self.loss_fn

    @tf.function
    def step(x, y):
      with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x, training=True))
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))

    self.step = step
    # Αρχική κατάσταση του optimizer (iterations = 0, learning_rate = 0.001, μηδενικά slots).
    # Επαναφέρεται ΑΚΡΙΒΩΣ πριν από κάθε νέο δίκτυο. ΔΙΟΡΘΩΣΗ: προηγουμένως όλες οι μεταβλητές
    # μηδενίζονταν, μαζί και το learning_rate, οπότε τα δίκτυα δεν εκπαιδεύονταν καθόλου.
    self.opt.build(self.model.trainable_variables)
    self._opt_init = [np.array(v.numpy(), copy=True) for v in self.opt.variables]

  @classmethod
  def get(cls, input_dim, num_classes):
    key = (input_dim, num_classes)
    if key not in cls._cache:
      cls._cache[key] = cls(input_dim, num_classes)
    return cls._cache[key]

  def _reinit(self):
    for layer in self.model.layers:
      if hasattr(layer, 'kernel_initializer'):
        layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        layer.bias.assign(layer.bias_initializer(layer.bias.shape))
    for v, v0 in zip(self.opt.variables, self._opt_init):
      v.assign(v0)

  def fit_predict_proba(self, X_tr, y_tr, X_te, epochs=30):
    self._reinit()
    x = tf.constant(X_tr, dtype=tf.float32)
    y = tf.constant(y_tr)
    for _ in range(epochs):
      self.step(x, y)
    return self.model(tf.constant(X_te, dtype=tf.float32), training=False).numpy()


def _project_pca(X_tr, X_te, R_k, n_pca):
  """Τυχαία προβολή -> PCA -> StandardScaler (fit μόνο στο train)."""
  S_tr_rp, S_te_rp = X_tr @ R_k, X_te @ R_k
  pca = PCA(n_components=n_pca, random_state=SEED)
  S_tr_pca, S_te_pca = pca.fit_transform(S_tr_rp), pca.transform(S_te_rp)
  sc = StandardScaler()
  return sc.fit_transform(S_tr_pca), sc.transform(S_te_pca)


def _pann_ec(X_tr, X_te, y_train, y_test, num_classes, d_rp, d_pca, seed):
  """Impl 1 / Impl 2: ένα MLP σε όλες τις προβολές μαζί (data augmentation), majority voting."""
  rng = np.random.default_rng(seed)
  p = X_tr.shape[1]
  X_aug, y_aug, X_te_list = [], [], []
  for _ in range(K_PROJECTIONS):
    R_k = rng.normal(0, 1 / np.sqrt(p), (p, d_rp))
    S_tr, S_te = _project_pca(X_tr, X_te, R_k, d_pca)
    X_aug.append(S_tr)
    y_aug.append(y_train)
    X_te_list.append(S_te)

  model = MLPClassifier(hidden_layer_sizes=(100, 64), max_iter=40, random_state=SEED, early_stopping=False)
  model.fit(np.vstack(X_aug), np.hstack(y_aug))
  preds_all = model.predict(np.vstack(X_te_list)).reshape(K_PROJECTIONS, len(y_test))
  y_pred = np.asarray(mode(preds_all, axis=0)[0]).flatten()
  return accuracy_score(y_test, y_pred), calc_f1(y_test, y_pred, num_classes)


def _multi_nn(X_tr, X_te, y_train, y_test, num_classes, d_rp, d_pca, seed, soft_voting):
  """Impl 3 (hard voting) / Impl 4 (soft voting): ένα Keras MLP ανά προβολή."""
  rng = np.random.default_rng(seed)
  tf.keras.utils.set_random_seed(seed)
  p = X_tr.shape[1]
  trainer = _FastTrainer.get(d_pca, num_classes)
  preds_all, probas_all = [], []
  for _ in range(K_PROJECTIONS):
    R_k = rng.normal(0, 1 / np.sqrt(p), (p, d_rp))
    S_tr, S_te = _project_pca(X_tr, X_te, R_k, d_pca)
    probs = trainer.fit_predict_proba(S_tr, y_train, S_te, epochs=30)
    probas_all.append(probs)
    preds_all.append(np.argmax(probs, axis=1))

  if soft_voting:
    y_pred = np.argmax(np.mean(probas_all, axis=0), axis=1)
  else:
    y_pred = np.asarray(mode(np.array(preds_all), axis=0)[0]).flatten()
  return accuracy_score(y_test, y_pred), calc_f1(y_test, y_pred, num_classes)


# Impl 1 & 3: raw χώρος, d_rp = 100, PCA 30 (όπως στο paper PANN-EC)
# Impl 2 & 4: ANOVA, d' από το config, d'' από τον D_DOUBLE_PRIME
def run_all_impls(X, y, tr, te, num_classes, cfg, split_seed, wanted=None):
  y_tr, y_te = y[tr], y[te]
  X_tr_raw, X_te_raw = preprocess(X[tr], X[te], y_tr, cfg['k_genes'], use_anova=False)
  X_tr_an, X_te_an = preprocess(X[tr], X[te], y_tr, cfg['k_genes'], use_anova=True)

  d_rp_raw = min(100, X_tr_raw.shape[1])
  d_pca_raw = min(30, d_rp_raw)
  d_rp_an = min(cfg['d_prime'], X_tr_an.shape[1])
  d_pca_an = min(D_DOUBLE_PRIME.get(cfg['display_name'], 30), d_rp_an)

  methods = {
      'Impl1_PANN_EC_Paper': (False, lambda: _pann_ec(X_tr_raw, X_te_raw, y_tr, y_te, num_classes, d_rp_raw, d_pca_raw, split_seed)),
      'Impl2_PANN_EC_ANOVA': (True, lambda: _pann_ec(X_tr_an, X_te_an, y_tr, y_te, num_classes, d_rp_an, d_pca_an, split_seed)),
      'Impl3_MultiNN_Hard': (False, lambda: _multi_nn(X_tr_raw, X_te_raw, y_tr, y_te, num_classes, d_rp_raw, d_pca_raw, split_seed, soft_voting=False)),
      'Impl4_MultiNN_Soft_ANOVA': (True, lambda: _multi_nn(X_tr_an, X_te_an, y_tr, y_te, num_classes, d_rp_an, d_pca_an, split_seed, soft_voting=True)),
  }
  out = []
  for name, (anova, fn) in methods.items():
    if wanted is not None and name not in wanted:
      continue
    t = timer()
    acc, f1 = fn()
    out.append({'Mode': name, 'ANOVA': anova, 'Accuracy': round(acc, 4), 'F1_Score': round(f1, 4),
                'Seconds': round(t(), 1)})
  return out


if __name__ == '__main__':
  print('    NEURAL BASELINES Impl 1-4 (Repeated Stratified CV, F1 score)\n', flush=True)

  # --impls 3     -> μόνο το Impl 3. ΠΡΟΣΟΧΗ: οι προβολές είναι ίδιες με το πλήρες τρέξιμο, αλλά η τυχαία
  #                  αρχικοποίηση των δικτύων εξαρτάται από την κατάσταση του TensorFlow (το Impl 4 τρέχει
  #                  πριν από το επόμενο split). Άρα τα F1 ΔΕΝ είναι ταυτόσημα με το πλήρες τρέξιμο
  #                  (επαληθεύτηκε), είναι όμως έγκυρη εκτέλεση της ίδιας μεθόδου.
  # --impls 1,3   -> Impl 1 και 3. Χωρίς την επιλογή: και τα 4.
  ALL_MODES = ['Impl1_PANN_EC_Paper', 'Impl2_PANN_EC_ANOVA', 'Impl3_MultiNN_Hard', 'Impl4_MultiNN_Soft_ANOVA']
  sel = [a.split('=', 1)[1] if a.startswith('--impls=') else sys.argv[i + 1]
         for i, a in enumerate(sys.argv) if a.startswith('--impls')]
  modes_wanted = [m for m in ALL_MODES if not sel or m[4] in sel[0].split(',')]
  print(f"    Μέθοδοι: {', '.join(modes_wanted)}\n", flush=True)

  done = set()
  if os.path.exists(OUT_PER_SPLIT):
    prev = pd.read_csv(OUT_PER_SPLIT)
    done = set(zip(prev['Dataset'], prev['Repeat'], prev['Fold'], prev['Mode']))
    print(f'    [RESUME] Βρέθηκαν {len(done)} ολοκληρωμένα (split, μέθοδος) — συνέχεια από εκεί.\n', flush=True)

  infos = []
  for cfg in DATASET_CONFIGS:
    d_name = cfg['display_name']
    print(f'>>> [BENCHMARKING DATASET]: {d_name.upper()}', flush=True)
    elapsed = timer()
    try:
      X, y, num_classes, info = load_dataset(cfg)
      infos.append(info)
      for rep, fold, split_seed, tr, te in iter_splits(y):
        todo = [m for m in modes_wanted if (d_name, rep, fold, m) not in done]
        if not todo:
          continue
        rows = run_all_impls(X, y, tr, te, num_classes, cfg, split_seed, wanted=set(todo))
        for r in rows:
          r.update({'Dataset': d_name, 'Repeat': rep, 'Fold': fold})
        df_split = pd.DataFrame(rows)[['Dataset', 'Repeat', 'Fold', 'Mode', 'ANOVA', 'Accuracy', 'F1_Score', 'Seconds']]
        df_split.to_csv(OUT_PER_SPLIT, mode='a', header=not os.path.exists(OUT_PER_SPLIT), index=False)
        print(f'    [rep {rep} fold {fold}] ' + ' | '.join(
            f"{r['Mode'].split('_')[0]}={r['F1_Score']:.3f} ({r['Seconds']:.0f}s)" for r in rows), flush=True)
      print(f'    --> Ολοκλήρωση {d_name} σε {elapsed():.1f}s\n', flush=True)
    except Exception as e:
      print(f'    [ΣΦΑΛΜΑ] {d_name}: {e}\n', flush=True)

  save_dataset_checks(infos)
  df = pd.read_csv(OUT_PER_SPLIT)
  per_dataset_summary(df, ['Mode']).to_csv(out_name('oneinall_results.csv'), index=False)

  pd.set_option('display.width', 1000)
  print('               ΣΥΓΚΕΝΤΡΩΤΙΚΑ Impl 1-4 (F1 score)')
  print(overall_summary(df, ['Mode']).to_string(index=False), flush=True)
