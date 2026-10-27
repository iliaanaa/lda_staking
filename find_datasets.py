"""
find_datasets.py — Ψάχνει στο OpenML τα ΣΩΣΤΑ datasets με βάση το αναμενόμενο N (δείγματα)
και p (γονίδια) από το common.py, και με βάση λέξεις-κλειδιά στο όνομα.

Δεν κατεβάζει δεδομένα — διαβάζει μόνο τη λίστα datasets του OpenML (ένα αρχείο JSON).
Έξοδος: dataset_candidates.csv  +  πίνακας στην οθόνη.
"""
import json
import urllib.error
import urllib.request

import pandas as pd

from common import DATASET_CONFIGS

API = 'https://www.openml.org/api/v1/json/data/list/status/all/limit/{limit}/offset/{offset}'
KEYWORDS = {
    'leukemia': ['leuk', 'allaml', 'all_aml', 'golub'],
    'colon': ['colon'],
    'prostate': ['prostate'],
    'SMK-CAN-187': ['smk'],
    'GLI-85': ['gli'],
    'dlbcl': ['dlbcl', 'lymphoma'],
    'ovarian': ['ovarian'],
    'CNS': ['cns', 'central', 'nervous', 'embryonal'],
    'Breast-Cancer': ['breast'],
    'Lung-Gordon': ['lung', 'gordon'],
}


def fetch_all_datasets():
  rows, offset, limit = [], 0, 10000
  while True:
    try:
      with urllib.request.urlopen(API.format(limit=limit, offset=offset), timeout=120) as r:
        batch = json.load(r)['data']['dataset']
    except urllib.error.HTTPError as e:
      if e.code == 412:   # το OpenML επιστρέφει 412 όταν δεν υπάρχουν άλλα αποτελέσματα
        break
      raise
    for d in batch:
      q = {item['name']: item['value'] for item in d.get('quality', [])}
      rows.append({
          'OpenML_ID': int(d['did']), 'Name': d['name'], 'Version': int(d.get('version', 1)),
          'Status': d.get('status', ''),
          'N': float(q.get('NumberOfInstances', 'nan')),
          'Columns': float(q.get('NumberOfFeatures', 'nan')),   # περιλαμβάνει και τη στήλη-στόχο
          'Classes': float(q.get('NumberOfClasses', 'nan')),
      })
    if len(batch) < limit:
      break
    offset += limit
  return pd.DataFrame(rows)


def search_by_name(allds, words):
  """Όλα τα datasets με >=1000 στήλες που περιέχουν κάποια από τις λέξεις στο όνομα."""
  hit = allds['Name'].str.lower().apply(lambda s: any(w.lower() in s for w in words))
  res = allds[hit & (allds['Columns'] >= 1000)].sort_values('Name')
  print(f'=== Αναζήτηση ονόματος: {words}  ({len(res)} αποτελέσματα)')
  print(res[['OpenML_ID', 'Name', 'Version', 'Status', 'N', 'Columns', 'Classes']].to_string(index=False))
  res.to_csv('name_search.csv', index=False)
  print('\nΑποθηκεύτηκε: name_search.csv')


if __name__ == '__main__':
  import sys
  print('Κατέβασμα λίστας datasets από το OpenML...', flush=True)
  allds = fetch_all_datasets()
  print(f'Βρέθηκαν {len(allds)} datasets.\n')
  if len(sys.argv) > 1:          # π.χ.  python find_datasets.py breast lung gordon
    search_by_name(allds, sys.argv[1:])
    sys.exit(0)

  out = []
  for cfg in DATASET_CONFIGS:
    name, exp_n, exp_p = cfg['display_name'], cfg['expected_n'], cfg['expected_p']
    kw = KEYWORDS.get(name, [])
    by_name = allds['Name'].str.lower().apply(lambda s: any(k in s for k in kw))
    by_size = (allds['N'] - exp_n).abs() <= 5
    cand = allds[(by_name | by_size) & (allds['Columns'] >= 1000)].copy()
    cand['Target'] = name
    cand['Expected_N'], cand['Expected_p'] = exp_n, exp_p
    cand['Name_Match'] = by_name[cand.index]
    cand['N_Match'] = (cand['N'] - exp_n).abs() <= 5
    cand['p_Match'] = ((cand['Columns'] - 1 - exp_p).abs() <= 0.01 * exp_p)
    cand['Score'] = cand['Name_Match'].astype(int) + 2 * cand['N_Match'].astype(int) + cand['p_Match'].astype(int)
    cand = cand.sort_values(['Score', 'Status'], ascending=[False, True]).head(8)
    out.append(cand)
    print(f'=== {name}  (αναμένεται N={exp_n}, p={exp_p})')
    if cand.empty:
      print('    Κανένας υποψήφιος στο OpenML.\n')
    else:
      print(cand[['OpenML_ID', 'Name', 'Version', 'Status', 'N', 'Columns', 'Classes',
                  'Name_Match', 'N_Match', 'p_Match']].to_string(index=False), '\n')

  pd.concat(out).to_csv('dataset_candidates.csv', index=False)
  print('Αποθηκεύτηκε: dataset_candidates.csv')
