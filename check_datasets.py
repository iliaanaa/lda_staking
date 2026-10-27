"""
check_datasets.py — ΤΡΕΞΕ ΤΟ ΠΡΩΤΟ. Φορτώνει κάθε dataset και δείχνει τι πραγματικά φορτώθηκε
(πηγή, N, p, κλάσεις, θετική κλάση για το F1 και με ποιον κανόνα επιλέχθηκε).

  python check_datasets.py                     -> τα 9 αρχικά          -> datasets_check.csv
  python check_datasets.py --datasets new      -> 7 OpenML + CuMiDa    -> datasets_check_new.csv
  (επίσης: --datasets openml_new | cumida | all)

Για τα νέα datasets εφαρμόζονται αυτόματα τα κριτήρια του PREREGISTRATION_new_datasets.md
(2 κλάσεις, N >= 40, μικρότερη κλάση >= 10, p >= 1000 και p > 5N). Στήλη 'Included'.
Τα αρχεία CuMiDa διαβάζονται από τον φάκελο ./cumida (ή από τη μεταβλητή περιβάλλοντος CUMIDA_DIR).
Το πρώτο διάβασμα κάθε CSV είναι αργό, μετά φορτώνεται από cache (cumida/_cache/*.npz).
"""
import pandas as pd

from common import DATASET_CONFIGS, DATASET_SET, DatasetExcluded, load_dataset, out_name, save_dataset_checks

if __name__ == '__main__':
  infos = []
  for cfg in DATASET_CONFIGS:
    print(f">>> {cfg['display_name']}", flush=True)
    try:
      infos.append(load_dataset(cfg)[3])
    except DatasetExcluded as e:
      print(f'    [ΑΠΟΚΛΕΙΣΜΟΣ] {e}')
      infos.append(e.info)
    except FileNotFoundError as e:
      print(f'    [ΛΕΙΠΕΙ] {e}')
      infos.append({'Dataset': cfg['display_name'], 'Group': cfg.get('group'), 'Check': 'ΛΕΙΠΕΙ ΤΟ ΑΡΧΕΙΟ',
                    'Included': 'ΟΧΙ: λείπει το αρχείο'})
    except Exception as e:
      print(f'    [ΣΦΑΛΜΑ] {e}')
      infos.append({'Dataset': cfg['display_name'], 'Group': cfg.get('group'), 'Check': f'ΣΦΑΛΜΑ: {e}',
                    'Included': 'ΟΧΙ: σφάλμα'})
    print()
  save_dataset_checks(infos)
  pd.set_option('display.width', 1000)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_colwidth', 60)
  cols = ['Dataset', 'Source', 'N', 'Expected_N', 'p', 'Expected_p', 'Class_Labels', 'Positive_Class_For_F1',
          'Positive_Rule', 'Check', 'Included']
  df = pd.DataFrame(infos).reindex(columns=cols)
  print(df.to_string(index=False))
  if DATASET_SET != 'original':
    inc = df['Included'].astype(str).str.startswith('ΝΑΙ').sum()
    print(f'\nΣυμμετέχουν: {inc} / {len(df)}')
    manual = df['Positive_Rule'].astype(str).str.contains('ΕΛΕΓΞΕ')
    if manual.any():
      print('Χρειάζονται χειροκίνητο έλεγχο θετικής κλάσης: ' + ', '.join(df.loc[manual, 'Dataset']))
  print(f"\nΑποθηκεύτηκε: {out_name('datasets_check.csv')}")
