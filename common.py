"""
common.py — Κοινός κώδικας για ΟΛΑ τα πειράματα LDA Stacking / Baselines.

Διορθώσεις σε σχέση με τις προηγούμενες εκδόσεις:
  1. Έλεγχος datasets: τυπώνεται το πραγματικό όνομα/ID του OpenML dataset που
     φορτώθηκε, το shape, η κατανομή κλάσεων και ποια κλάση είναι η "θετική" για το F1.
     Αν το N ή το p δεν συμφωνεί με τις αναμενόμενες τιμές, εμφανίζεται ΠΡΟΕΙΔΟΠΟΙΗΣΗ.
  2. Μία μετρική παντού: calc_f1 (binary F1 με pos_label=1 για 2 κλάσεις,
     weighted F1 για >2 κλάσεις) — στην αξιολόγηση, στη βαθμολόγηση προβολών
     και στη ρύθμιση κατωφλίου.
  3. Repeated Stratified K-Fold (5 folds x N_REPEATS) αντί για ένα 80/20 split.
     Κάθε fold έχει 20% test, όπως πριν. Όλα τα scripts χρησιμοποιούν ΤΑ ΙΔΙΑ splits.
     Οι τυχαίες προβολές παράγονται με τοπικό rng (np.random.default_rng(seed_του_split)),
     άρα κάθε στρατηγική/mode βλέπει ΤΙΣ ΙΔΙΕΣ προβολές και τα αποτελέσματα
     αναπαράγονται ακριβώς σε κάθε τρέξιμο.
"""
import os
import re
import socket
import sys
import time
import warnings

import numpy as np
import pandas as pd

socket.setdefaulttimeout(120.0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΙΚΟΥ ΠΡΩΤΟΚΟΛΛΟΥ
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5      # 5 folds => κάθε test fold = 20% (όπως το παλιό 80/20)
N_REPEATS = 2     # 5 x 2 = 10 αξιολογήσεις ανά dataset. Αύξησέ το για πιο σταθερά αποτελέσματα.
N_JOBS = -1       # παραλληλισμός στο cross_val_predict
THRESH_GRID = np.linspace(0.1, 0.9, 81)
STRICT_DATASET_CHECK = True   # σταματά αν το dataset που φορτώθηκε δεν έχει το αναμενόμενο N / p

# expected_n / expected_p: τιμές από τις σημειώσεις του project.
# Χρησιμοποιούνται ΜΟΝΟ για έλεγχο — αν το dataset που φορτώθηκε διαφέρει, βγαίνει προειδοποίηση.
ORIGINAL_CONFIGS = [
    # Ένα μόνο OpenML ID ανά dataset (επιβεβαιωμένα με find_datasets.py: σωστό όνομα, N και p).
    # ΧΩΡΙΣ εναλλακτικά IDs: αυτά έφερναν σιωπηλά λάθος datasets.
    {'display_name': 'leukemia', 'identifiers': [45090], 'k_genes': 50, 'd_prime': 100, 'expected_n': 72, 'expected_p': 7129, 'positive_label': 'AML'},  # ALL vs AML: δύο λευχαιμίες, θετική η AML (μειοψηφία)
    {'display_name': 'colon', 'identifiers': [45087], 'k_genes': 100, 'd_prime': 100, 'expected_n': 62, 'expected_p': 2000, 'positive_label': '-1'},  # Alon: 40 όγκοι / 22 υγιή -> '-1'(40) = όγκος
    {'display_name': 'prostate', 'identifiers': [45099], 'k_genes': 50, 'd_prime': 100, 'expected_n': 102, 'expected_p': 12600, 'positive_label': '1'},  # Singh: 52 όγκοι / 50 υγιή -> '1'(52) = όγκος
    {'display_name': 'SMK-CAN-187', 'identifiers': [45100], 'k_genes': 500, 'd_prime': 100, 'expected_n': 187, 'expected_p': 19993, 'positive_label': '2'},  # 90 control / 97 tumor -> '2'(97) = καρκίνος
    {'display_name': 'GLI-85', 'identifiers': [45089], 'k_genes': 200, 'd_prime': 100, 'expected_n': 85, 'expected_p': 22283, 'positive_label': '2'},  # Freije: γλοιώματα grade III / IV. Πιθανότατα '2'(59) = grade IV (μη επιβεβαιωμένο)
    {'display_name': 'dlbcl', 'identifiers': [45088], 'k_genes': 50, 'd_prime': 100, 'expected_n': 77, 'expected_p': 5469, 'positive_label': '0'},  # Shipp: 58 DLBCL / 19 FL -> '0'(58) = DLBCL (επιθετικότερο)
    {'display_name': 'ovarian', 'identifiers': [45098], 'k_genes': 500, 'd_prime': 100, 'expected_n': 253, 'expected_p': 15154, 'positive_label': 'Cancer'},
    {'display_name': 'CNS', 'identifiers': [45086], 'k_genes': 200, 'd_prime': 100, 'expected_n': 60, 'expected_p': 7129, 'positive_label': '0'},  # Pomeroy: Class0 = 39 αποτυχίες θεραπείας, Class1 = 21 επιζώντες (επιβεβαιωμένο)
    {'display_name': 'Breast', 'identifiers': [45085], 'k_genes': 300, 'd_prime': 100, 'expected_n': 97, 'expected_p': 24481, 'positive_label': 'relapse'},  # υποτροπή = θετική
    # Τα αρχικά Breast-Cancer (286x22283) και Lung-Gordon (181x12533) δεν υπάρχουν στο OpenML.
]


# ---------------------------------------------------------------------------
# ΝΕΑ DATASETS (PREREGISTRATION_new_datasets.md) — ΔΕΝ αγγίζουν τα 9 αρχικά
# ---------------------------------------------------------------------------
# k_genes για ΟΛΑ τα νέα datasets = 200 = η διάμεσος των k των 9 αρχικών
# (50, 50, 50, 100, 200, 200, 300, 500, 500). Ορίστηκε πριν τρέξει οτιδήποτε στα νέα δεδομένα.
NEW_K_GENES = 200

# Κριτήρια αποκλεισμού (ελέγχονται αυτόματα στη φόρτωση με τις ΑΚΡΙΒΕΙΣ μετρήσεις)
SCREEN_MIN_N = 40
SCREEN_MIN_CLASS = 10
SCREEN_MIN_P = 1000
SCREEN_P_OVER_N = 5

NEW_OPENML_CONFIGS = [
    {'display_name': 'lymphoma_2classes', 'identifiers': [1101], 'expected_n': 45, 'expected_p': 4026, 'positive_label': 'ACL'},
    {'display_name': 'armstrong-2002-v1', 'identifiers': [46775], 'expected_n': 72, 'expected_p': 1081, 'positive_label': 'MLL'},
    {'display_name': 'anthracyclineTaxane', 'identifiers': [1085], 'expected_n': 159, 'expected_p': 61359, 'positive_label': 'minority'},
    {'display_name': 'AP_Colon_Ovary', 'identifiers': [1153], 'expected_n': 484, 'expected_p': None, 'positive_label': 'minority'},
    {'display_name': 'AP_Endometrium_Breast', 'identifiers': [1123], 'expected_n': 405, 'expected_p': None, 'positive_label': 'minority'},
    {'display_name': 'AP_Omentum_Kidney', 'identifiers': [1147], 'expected_n': 337, 'expected_p': None, 'positive_label': 'minority'},
    {'display_name': 'AP_Prostate_Uterus', 'identifiers': [1131], 'expected_n': 193, 'expected_p': None, 'positive_label': 'minority'},
]
for _c in NEW_OPENML_CONFIGS:
  _c.update({'k_genes': NEW_K_GENES, 'd_prime': 100, 'screen': True, 'group': 'openml_new'})

# CuMiDa (Feltes et al. 2019). Τοπικά CSV: στήλη 'samples', στήλη 'type' (κλάση), μετά τα γονίδια.
# Ο φάκελος ορίζεται με τη μεταβλητή περιβάλλοντος CUMIDA_DIR (προεπιλογή: ./cumida).
CUMIDA_DIR = os.environ.get('CUMIDA_DIR', 'cumida')
# (αρχείο, αναμενόμενο N, αναμενόμενο p) — από τον πίνακα του CuMiDa. Head/Neck αρχεία = 'Throat_*'.
CUMIDA_FILES = [
    ('Liver_GSE14520_U133A', 357, 22277), ('Pancreatic_GSE16515', 51, 54675), ('Lung_GSE18842', 90, 54675),
    ('Lung_GSE19804', 114, 54675), ('Liver_GSE22405', 48, 22283), ('Leukemia_GSE22529_U133A', 52, 22283),
    ('Breast_GSE22820', 139, 33579), ('Colorectal_GSE25070', 52, 24526), ('Lung_GSE27262', 48, 54675),
    ('Bladder_GSE31189', 85, 54675), ('Leukemia_GSE33615', 71, 33579), ('Breast_GSE38959', 43, 33579),
    ('Breast_GSE42568', 116, 54675), ('Throat_GSE42743', 103, 54675), ('Colorectal_GSE44076', 194, 49386),
    ('Colorectal_GSE44861', 105, 22277), ('Prostate_GSE46602', 49, 54675), ('Liver_GSE50579', 76, 36547),
    ('Renal_GSE53757', 143, 54675), ('Liver_GSE57957', 75, 47323), ('Throat_GSE59102', 42, 32703),
    ('Breast_GSE59246', 101, 36622), ('Liver_GSE62232', 91, 54675), ('Leukemia_GSE63270', 101, 54675),
    ('Lung_GSE63459', 65, 24526), ('Prostate_GSE6919_U95Av2', 124, 12625), ('Breast_GSE70947', 289, 35981),
    ('Liver_GSE76427', 165, 47322), ('Lung_GSE7670', 51, 22283), ('Colorectal_GSE8671', 63, 54675),
]
CUMIDA_CONFIGS = [
    {'display_name': f'CuMiDa_{name}', 'cumida_file': f'{name}.csv', 'expected_n': n, 'expected_p': p,
     'positive_label': 'auto_normal', 'k_genes': NEW_K_GENES, 'd_prime': 100, 'screen': True, 'group': 'cumida'}
    for name, n, p in CUMIDA_FILES
]
# Όχι σύγκριση με φυσιολογικό ιστό -> κανόνας 4: η κλινικά σοβαρότερη κατάσταση.
# Breast_GSE59246: διηθητικός καρκίνος (IBC) έναντι in situ (DCIS) -> θετική η IBC.
CUMIDA_POSITIVE_OVERRIDE = {'Breast_GSE59246': 'IBC'}
for _c in CUMIDA_CONFIGS:
  _key = _c['cumida_file'][:-4]
  if _key in CUMIDA_POSITIVE_OVERRIDE:
    _c['positive_label'] = CUMIDA_POSITIVE_OVERRIDE[_key]
NORMAL_PATTERN = re.compile(r'normal|control|healthy|non[-_ ]?tumou?r', re.I)

# ---------------------------------------------------------------------------
# ΕΠΙΛΟΓΗ ΣΥΝΟΛΟΥ DATASETS:   python <script>.py --datasets original|new|openml_new|cumida|all
# Προεπιλογή: original (τα 9 αρχικά — όλα τα προηγούμενα αποτελέσματα μένουν ως έχουν).
# Για οτιδήποτε άλλο, τα αρχεία εξόδου παίρνουν κατάληξη, π.χ. baselines_comparison_results_new.csv
# ---------------------------------------------------------------------------
DATASET_SETS = {
    'original': ORIGINAL_CONFIGS,
    'openml_new': NEW_OPENML_CONFIGS,
    'cumida': CUMIDA_CONFIGS,
    'new': NEW_OPENML_CONFIGS + CUMIDA_CONFIGS,
    'all': ORIGINAL_CONFIGS + NEW_OPENML_CONFIGS + CUMIDA_CONFIGS,
}


def _parse_dataset_set(argv):
  for i, a in enumerate(argv):
    if a == '--datasets' and i + 1 < len(argv):
      return argv[i + 1]
    if a.startswith('--datasets='):
      return a.split('=', 1)[1]
  return 'original'


DATASET_SET = _parse_dataset_set(sys.argv)
if DATASET_SET not in DATASET_SETS:
  raise SystemExit(f"Άγνωστο --datasets {DATASET_SET!r}. Επιλογές: {', '.join(DATASET_SETS)}")
DATASET_CONFIGS = DATASET_SETS[DATASET_SET]
SET_SUFFIX = '' if DATASET_SET == 'original' else f'_{DATASET_SET}'


def out_name(filename):
  """'x.csv' -> 'x_new.csv' όταν τρέχει άλλο σύνολο από τα αρχικά (ώστε να μη σβηστούν τα παλιά αποτελέσματα)."""
  base, ext = os.path.splitext(filename)
  return f'{base}{SET_SUFFIX}{ext}'


class DatasetExcluded(Exception):
  """Το dataset αποκλείεται από τα προκαθορισμένα κριτήρια (όχι σφάλμα φόρτωσης)."""


# ---------------------------------------------------------------------------
# ΦΟΡΤΩΣΗ & ΕΛΕΓΧΟΣ DATASET (Διόρθωση 1)
# ---------------------------------------------------------------------------
def fetch_microarray_robust(identifiers):
  """Επιστρέφει (data, identifier_που_πέτυχε). Οι εναλλακτικοί identifiers μπορεί να
  αντιστοιχούν σε ΔΙΑΦΟΡΕΤΙΚΑ datasets — γι' αυτό καταγράφουμε ποιος χρησιμοποιήθηκε."""
  last_err = None
  for identifier in identifiers:
    for parser_type in ['auto', 'liac-arff']:
      try:
        if isinstance(identifier, int):
          return fetch_openml(data_id=identifier, as_frame=True, parser=parser_type), identifier
        return fetch_openml(name=identifier, version=1, as_frame=True, parser=parser_type), identifier
      except Exception as e:
        last_err = e
        continue
  raise ValueError(f'Αδυναμία φόρτωσης: {identifiers} (τελευταίο σφάλμα: {type(last_err).__name__}: {last_err})')


def _load_openml_raw(config):
  data, used_id = fetch_microarray_robust(config['identifiers'])
  df_x = data.data.copy()
  for col in df_x.columns:
    df_x[col] = pd.to_numeric(df_x[col], errors='coerce')
  df_x = df_x.dropna(axis=1, how='all')
  details = getattr(data, 'details', None) or {}
  meta = {'Source': 'OpenML', 'Loaded_With': used_id,
          'OpenML_Name': details.get('name', '?'), 'OpenML_ID': details.get('id', '?')}
  # Κάποια ARFF έχουν ετικέτες με εισαγωγικά (π.χ. armstrong: '"ALL"') -> αφαιρούνται
  labels = np.array([str(v).strip().strip('\'"').strip() for v in data.target], dtype=str)
  return df_x.values.astype(float), labels, meta


def _load_cumida_raw(config):
  """Διαβάζει CUMIDA_DIR/<αρχείο>.csv. Κρατά cache (.npz) για γρήγορη επαναφόρτωση."""
  path = os.path.join(CUMIDA_DIR, config['cumida_file'])
  cache = os.path.join(CUMIDA_DIR, '_cache', config['cumida_file'].replace('.csv', '.npz'))
  if not os.path.exists(path) and not os.path.exists(cache):
    raise FileNotFoundError(f'Λείπει το αρχείο {path}')
  if os.path.exists(cache) and (not os.path.exists(path) or os.path.getmtime(cache) >= os.path.getmtime(path)):
    z = np.load(cache, allow_pickle=False)
    X, labels, n_dup = z['X'], z['labels'], int(z['n_dup'])
  else:
    df = pd.read_csv(path)
    if 'type' not in df.columns:
      raise ValueError(f"{path}: δεν υπάρχει στήλη 'type'")
    labels = np.array([str(v).strip() for v in df['type']], dtype=str)
    n_dup = int(df['samples'].duplicated().sum()) if 'samples' in df.columns else 0
    df_x = df.drop(columns=[c for c in ('samples', 'type') if c in df.columns])
    df_x = df_x.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    X = df_x.values.astype(float)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, X=X, labels=labels, n_dup=n_dup)
  meta = {'Source': 'CuMiDa', 'Loaded_With': config['cumida_file'], 'OpenML_Name': '-', 'OpenML_ID': '-',
          'Duplicate_Sample_IDs': n_dup}
  return X, np.asarray(labels).astype(str), meta


def _choose_positive(config, classes, counts):
  """Επιστρέφει (θετική κλάση, κανόνας). Κανόνες του PREREGISTRATION_new_datasets.md."""
  rule = config.get('positive_label')
  str_classes = [str(c) for c in classes]
  minority = str_classes[int(np.argmin(counts))] if counts[0] != counts[1] else str_classes[1]
  if rule is None:
    return str_classes[1], 'default (2η κλάση)'
  if rule == 'minority':
    return minority, 'μειοψηφία'
  if rule == 'auto_normal':
    normal = [c for c in str_classes if NORMAL_PATTERN.search(c)]
    if len(normal) == 1:
      return [c for c in str_classes if c != normal[0]][0], f"ασθένεια έναντι '{normal[0]}'"
    return minority, 'μειοψηφία (δεν βρέθηκε μία φυσιολογική κλάση: ΕΛΕΓΞΕ ΧΕΙΡΟΚΙΝΗΤΑ)'
  if str(rule) not in str_classes:
    raise ValueError(f"positive_label={rule!r} δεν υπάρχει στις κλάσεις {str_classes} ({config['display_name']})")
  return str(rule), 'ρητή'


def load_dataset(config, verbose=True):
  """Φορτώνει ΟΛΟ το dataset (χωρίς split). Επιστρέφει X, y, num_classes, info.
  Για τα νέα datasets ('screen': True) εφαρμόζει και τα κριτήρια αποκλεισμού (DatasetExcluded)."""
  if 'cumida_file' in config:
    X, target_values, meta = _load_cumida_raw(config)
  else:
    X, target_values, meta = _load_openml_raw(config)

  classes = np.unique(target_values)
  y = np.array([np.where(classes == label)[0][0] for label in target_values])

  unique_classes, class_counts = np.unique(y, return_counts=True)
  invalid_classes = unique_classes[class_counts < 2]
  if len(invalid_classes) > 0:
    valid_mask = ~np.isin(y, invalid_classes)
    X, y = X[valid_mask], y[valid_mask]
    kept = classes[np.unique(y)]
    classes = kept
    y = np.array([np.where(np.unique(y) == label)[0][0] for label in y])

  # Θετική κλάση για το binary F1 -> μεταφέρεται στον δείκτη 1, ώστε calc_f1(pos_label=1) να τη μετράει.
  pos_rule = '-'
  if len(classes) == 2:
    pos, pos_rule = _choose_positive(config, classes, np.bincount(y))
    if [str(c) for c in classes].index(pos) == 0:
      classes = classes[::-1]
      y = 1 - y

  counts = np.bincount(y)
  info = {
      'Dataset': config['display_name'],
      'Group': config.get('group', 'original'),
      **meta,
      'N': X.shape[0],
      'p': X.shape[1],
      'Expected_N': config.get('expected_n'),
      'Expected_p': config.get('expected_p'),
      'Num_Classes': len(classes),
      'Class_Labels': ' | '.join(f'{c}={n}' for c, n in zip(classes, counts)),
      'Minority_Size': int(counts.min()),
      'Positive_Class_For_F1': str(classes[1]) if len(classes) == 2 else 'weighted (multiclass)',
      'Positive_Rule': pos_rule,
      'Missing_Values': int(np.isnan(X).sum()),
  }
  problems = []
  if info['Expected_N'] is not None and info['N'] != info['Expected_N']:
    problems.append(f"N={info['N']} αντί για {info['Expected_N']}")
  if info['Expected_p'] is not None and abs(info['p'] - info['Expected_p']) > 0.01 * info['Expected_p']:
    problems.append(f"p={info['p']} αντί για {info['Expected_p']}")
  info['Check'] = 'OK' if not problems else 'ΠΡΟΣΟΧΗ: ' + '; '.join(problems)

  # Κριτήρια αποκλεισμού (μόνο για τα νέα datasets)
  excl = []
  if config.get('screen'):
    if len(classes) != 2:
      excl.append(f'{len(classes)} κλάσεις (απαιτούνται 2)')
    if info['N'] < SCREEN_MIN_N:
      excl.append(f"N={info['N']} < {SCREEN_MIN_N}")
    if info['Minority_Size'] < SCREEN_MIN_CLASS:
      excl.append(f"μικρότερη κλάση={info['Minority_Size']} < {SCREEN_MIN_CLASS}")
    if info['p'] < SCREEN_MIN_P or info['p'] <= SCREEN_P_OVER_N * info['N']:
      excl.append(f"p={info['p']} (απαιτείται p ≥ {SCREEN_MIN_P} και p > {SCREEN_P_OVER_N}N)")
  info['Included'] = 'ΝΑΙ' if not excl else 'ΟΧΙ: ' + '; '.join(excl)

  if verbose:
    src = (f"OpenML '{info['OpenML_Name']}' (id={info['OpenML_ID']})" if meta['Source'] == 'OpenML'
           else f"CuMiDa '{meta['Loaded_With']}'")
    print(f"    [DATA] {src}")
    print(f"    [DATA] N={info['N']}, p={info['p']}, κλάσεις: {info['Class_Labels']}")
    print(f"    [DATA] Θετική κλάση για το F1: {info['Positive_Class_For_F1']}  (κανόνας: {pos_rule})")
    print(f"    [DATA] Έλεγχος: {info['Check']}   Συμμετοχή: {info['Included']}", flush=True)

  if problems and STRICT_DATASET_CHECK:
    raise ValueError(f"Λάθος dataset για '{config['display_name']}' ({meta['Loaded_With']}): " + '; '.join(problems))
  if excl:
    exc = DatasetExcluded(f"Αποκλείεται '{config['display_name']}': " + '; '.join(excl))
    exc.info = info
    raise exc
  return X, y, len(classes), info


# ---------------------------------------------------------------------------
# SPLITS (Διόρθωση 3) — ίδια για όλα τα scripts
# ---------------------------------------------------------------------------
def iter_splits(y):
  """Repeated Stratified K-Fold. Yields (repeat, fold, split_seed, train_idx, test_idx)."""
  min_count = int(np.bincount(y).min())
  n_splits = max(2, min(N_SPLITS, min_count))
  if n_splits < N_SPLITS:
    print(f'    [WARN] Η μικρότερη κλάση έχει {min_count} δείγματα -> {n_splits} folds', flush=True)
  rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=SEED)
  for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
    yield i // n_splits, i % n_splits, SEED + i, tr, te


def preprocess(X_tr_raw, X_te_raw, y_train, k_genes, use_anova=True):
  """Imputation -> (ANOVA) -> Scaling. Όλα fit ΜΟΝΟ στο train fold."""
  imputer = SimpleImputer(strategy='mean')
  X_tr = imputer.fit_transform(X_tr_raw)
  X_te = imputer.transform(X_te_raw)
  if use_anova:
    selector = SelectKBest(score_func=f_classif, k=min(k_genes, X_tr.shape[1]))
    X_tr = selector.fit_transform(X_tr, y_train)
    X_te = selector.transform(X_te)
  scaler = StandardScaler()
  return scaler.fit_transform(X_tr), scaler.transform(X_te)


# ---------------------------------------------------------------------------
# ΜΕΤΡΙΚΗ (Διόρθωση 2) — F1 score παντού
# ---------------------------------------------------------------------------
def calc_f1(y_true, y_pred, num_classes):
  if num_classes == 2:
    return f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
  return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def tune_threshold(scores, y_true, num_classes, thresh_range=THRESH_GRID):
  """Επιστρέφει (best_threshold, best_f1) πάνω σε OOF scores του train set."""
  best_t, best_f1 = thresh_range[0], -1.0
  for t in thresh_range:
    f1_t = calc_f1(y_true, (scores >= t).astype(int), num_classes)
    if f1_t > best_f1:
      best_f1, best_t = f1_t, t
  return best_t, best_f1


# ---------------------------------------------------------------------------
# ΤΥΧΑΙΕΣ ΠΡΟΒΟΛΕΣ + BASE LDA
# ---------------------------------------------------------------------------
def _pad(p, num_classes):
  if p.shape[1] < num_classes:
    fixed = np.zeros((p.shape[0], num_classes))
    fixed[:, :p.shape[1]] = p
    return fixed
  return p


def generate_projections(X_tr, X_te, y_train, num_classes, d_prime, k_proj, seed):
  """Παράγει k_proj προβολές με ΤΟΠΙΚΟ rng(seed): ίδιες προβολές για όλες τις στρατηγικές."""
  rng = np.random.default_rng(seed)
  curr_d = X_tr.shape[1]
  actual_d_prime = min(d_prime, curr_d)
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
  proj_train, proj_test, proj_scores = [], [], []
  for _ in range(k_proj):
    R_i = rng.normal(0, 1 / np.sqrt(curr_d), (curr_d, actual_d_prime))
    S_tr, S_te = X_tr @ R_i, X_te @ R_i
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    oof_p = _pad(cross_val_predict(clf, S_tr, y_train, cv=cv, method='predict_proba', n_jobs=N_JOBS), num_classes)
    clf.fit(S_tr, y_train)
    te_p = _pad(clf.predict_proba(S_te), num_classes)
    proj_train.append(oof_p)
    proj_test.append(te_p)
    proj_scores.append(calc_f1(y_train, np.argmax(oof_p, axis=1), num_classes))
  return proj_train, proj_test, np.array(proj_scores)


# ---------------------------------------------------------------------------
# ΕΠΙΛΟΓΗ ΥΠΟΧΩΡΩΝ
# ---------------------------------------------------------------------------
LAMBDA_DIV = 0.25


def select_subspaces(mode, proj_train, proj_scores, num_classes, n_selected, seed=SEED, lam=None):
  """mode: 'top_f1' | 'mrmr' | 'diversity' | 'random' | 'all' | 'l1_sparse'.
  Ίδια λογική με τις παλιές εκδόσεις. Στα mrmr/diversity ο μέσος όρος πάνω στους ήδη
  επιλεγμένους κρατιέται σε αθροιστική μορφή (πολύ ταχύτερο, ίδια επιλογή)."""
  k_proj = len(proj_scores)
  n_sel = min(n_selected, k_proj)
  scores = np.asarray(proj_scores, dtype=float)

  if mode in ('all', 'l1_sparse'):
    return np.arange(k_proj)
  if mode == 'top_f1':
    return np.argsort(scores)[-n_sel:]
  if mode == 'random':
    return np.random.default_rng(seed).choice(k_proj, size=n_sel, replace=False)

  if mode == 'mrmr':
    probs = np.column_stack([p[:, 1] if num_classes == 2 else p[:, 0] for p in proj_train])
    with np.errstate(invalid='ignore', divide='ignore'):
      pair_matrix = np.nan_to_num(np.abs(np.corrcoef(probs, rowvar=False)), nan=1.0)
    sign = -1.0   # relevance - redundancy
  elif mode == 'diversity':
    preds = np.column_stack([np.argmax(p, axis=1) for p in proj_train])
    pair_matrix = None
    sign = LAMBDA_DIV if lam is None else lam   # relevance + lambda * diversity
  else:
    raise ValueError(f'Άγνωστο mode: {mode}')

  selected = [int(np.argmax(scores))]
  available = np.ones(k_proj, dtype=bool)
  available[selected[0]] = False
  disagree_count = np.zeros(k_proj, dtype=np.int64)
  n_samples = preds.shape[0] if pair_matrix is None else None
  while len(selected) < n_sel:
    if mode == 'mrmr':
      term = np.mean(pair_matrix[:, selected], axis=1)
    else:
      # ακέραιος αθροιστικός μετρητής διαφωνιών -> ίδιο αποτέλεσμα με τον αρχικό κώδικα, πολύ ταχύτερα
      disagree_count += np.sum(preds != preds[:, [selected[-1]]], axis=0)
      term = disagree_count / (n_samples * len(selected))
    total = scores + sign * term
    total[~available] = -np.inf
    best = int(np.argmax(total))
    selected.append(best)
    available[best] = False
  return np.array(selected)


# ---------------------------------------------------------------------------
# META-LEARNER + ΚΑΤΩΦΛΙ
# ---------------------------------------------------------------------------
def meta_stack(proj_train, proj_test, selected_idx, y_train, y_test, num_classes, l1=False):
  """Επιστρέφει dict με Accuracy, F1_Score (test) και Train_OOF_F1 (για επιλογή K,M χωρίς test)."""
  P_tr = np.hstack([proj_train[i] for i in selected_idx])
  P_te = np.hstack([proj_test[i] for i in selected_idx])
  if l1:
    meta = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', max_iter=500, random_state=SEED)
  else:
    meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=SEED)

  cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
  meta_oof = cross_val_predict(meta, P_tr, y_train, cv=cv_meta, method='predict_proba', n_jobs=N_JOBS)
  if num_classes == 2:
    best_t, train_oof_f1 = tune_threshold(meta_oof[:, 1], y_train, num_classes)
  else:
    best_t, train_oof_f1 = None, calc_f1(y_train, np.argmax(meta_oof, axis=1), num_classes)

  meta.fit(P_tr, y_train)
  test_p = meta.predict_proba(P_te)
  y_pred = (test_p[:, 1] >= best_t).astype(int) if num_classes == 2 else np.argmax(test_p, axis=1)
  return {
      'Accuracy': accuracy_score(y_test, y_pred),
      'F1_Score': calc_f1(y_test, y_pred, num_classes),
      'Train_OOF_F1': train_oof_f1,
  }


# ---------------------------------------------------------------------------
# ΣΥΓΚΕΝΤΡΩΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
# ---------------------------------------------------------------------------
def per_dataset_summary(df, group_cols, value_cols=('Accuracy', 'F1_Score')):
  """Μέσος όρος & τυπική απόκλιση ΠΑΝΩ ΣΤΑ SPLITS, ανά dataset/μέθοδο."""
  agg = df.groupby(['Dataset'] + list(group_cols), sort=False)[list(value_cols)].agg(['mean', 'std'])
  agg.columns = [f'{v}_{s}' for v, s in agg.columns]
  return agg.reset_index().round(4)


def overall_summary(df, group_cols, value='F1_Score'):
  """Πρώτα μέσος όρος ανά dataset (πάνω στα splits), μετά μέσος/std πάνω στα datasets."""
  per_ds = df.groupby(['Dataset'] + list(group_cols), sort=False)[value].mean().reset_index()
  out = per_ds.groupby(list(group_cols), sort=False)[value].agg(['mean', 'std']).reset_index()
  out.columns = list(group_cols) + [f'Mean {value} (across datasets)', 'Std across datasets']
  return out.round(4)


def wide_summary(df, label_map):
  """Μορφή όπως τα παλιά CSV: μία γραμμή ανά dataset, στήλες <label>_Acc, <label>_F1 (μέσοι όροι
  πάνω στα splits) και <label>_F1_std."""
  rows = []
  for d_name, g in df.groupby('Dataset', sort=False):
    row = {'Dataset': d_name}
    for mode, label in label_map.items():
      gm = g[g['Mode'] == mode]
      row[f'{label}_Acc'] = round(gm['Accuracy'].mean(), 4)
      row[f'{label}_F1'] = round(gm['F1_Score'].mean(), 4)
      row[f'{label}_F1_std'] = round(gm['F1_Score'].std(), 4)
    rows.append(row)
  return pd.DataFrame(rows)


def run_mode_comparison(modes, use_anova, k_proj, n_selected):
  """Για κάθε dataset & split: παράγει k_proj προβολές ΜΙΑ φορά και αξιολογεί όλα τα modes
  πάνω στις ΙΔΙΕΣ προβολές. modes: dict {όνομα_mode: (selection_mode, l1_meta)}."""
  results, infos = [], []
  for cfg in DATASET_CONFIGS:
    d_name = cfg['display_name']
    print(f'>>> [BENCHMARKING DATASET]: {d_name.upper()}  (ANOVA={use_anova})', flush=True)
    elapsed = timer()
    try:
      X, y, num_classes, info = load_dataset(cfg)
      infos.append(info)
      for rep, fold, split_seed, tr, te in iter_splits(y):
        X_tr, X_te = preprocess(X[tr], X[te], y[tr], cfg['k_genes'], use_anova)
        p_tr, p_te, p_sc = generate_projections(X_tr, X_te, y[tr], num_classes, cfg['d_prime'], k_proj, seed=split_seed)
        line = []
        for mode_name, (sel_mode, l1) in modes.items():
          idx = select_subspaces(sel_mode, p_tr, p_sc, num_classes, n_selected, seed=split_seed)
          res = meta_stack(p_tr, p_te, idx, y[tr], y[te], num_classes, l1=l1)
          results.append({'Dataset': d_name, 'ANOVA': use_anova, 'Repeat': rep, 'Fold': fold, 'Mode': mode_name,
                          'Accuracy': round(res['Accuracy'], 4), 'F1_Score': round(res['F1_Score'], 4),
                          'Train_OOF_F1': round(res['Train_OOF_F1'], 4)})
          line.append(f"{mode_name}={res['F1_Score']:.4f}")
        print(f'    [rep {rep} fold {fold}] ' + ' | '.join(line), flush=True)
      print(f'    --> Total Dataset Time: {elapsed():.1f}s\n', flush=True)
    except Exception as e:
      print(f'    [ΣΦΑΛΜΑ] {d_name}: {e}\n', flush=True)
  return pd.DataFrame(results), infos


def save_dataset_checks(infos, path='datasets_check.csv'):
  pd.DataFrame(infos).to_csv(out_name(path), index=False)


def timer():
  t0 = time.time()
  return lambda: time.time() - t0
