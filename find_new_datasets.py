"""
find_new_datasets.py — Αναζήτηση ΝΕΩΝ υψηλοδιάστατων datasets στο OpenML για την τελική σύγκριση.

ΔΕΝ κατεβάζει τα δεδομένα και ΔΕΝ τρέχει κανένα μοντέλο. Διαβάζει μόνο:
  1. τη λίστα όλων των datasets του OpenML (μέγεθος, κλάσεις),
  2. για τους υποψήφιους: περιγραφή, αναφορά (citation) και τις ετικέτες της κλάσης-στόχου.
Έτσι η επιλογή γίνεται ΧΩΡΙΣ να δούμε αποτελέσματα μοντέλων.

Κριτήρια (αυτόματα, βήμα 1):
  - 2 κλάσεις
  - p >= 1.000 χαρακτηριστικά και p > 5 x N
  - N >= 40 και μικρότερη κλάση >= 10 δείγματα
  - όχι τα 9 datasets που ήδη χρησιμοποιούμε (ούτε πιθανά διπλότυπά τους, ίδιο N και p)
  - εξαιρούνται προφανώς μη βιοϊατρικά (QSAR χημικές ενώσεις, κείμενα κ.λπ.)
Τα υπόλοιπα κριτήρια (omics δεδομένα, θετική κλάση, κοινά δείγματα) ελέγχονται χειροκίνητα
με βάση το new_dataset_candidates_details.txt.

Εκτέλεση:  python find_new_datasets.py
Έξοδοι:
  new_dataset_candidates.csv          -> όλοι οι υποψήφιοι με τα βασικά στοιχεία
  new_dataset_candidates_details.txt  -> περιγραφή, αναφορά, ετικέτες κλάσης ανά υποψήφιο
"""
import json
import re
import time
import urllib.error
import urllib.request

import pandas as pd

from common import ORIGINAL_CONFIGS as DATASET_CONFIGS

API = 'https://www.openml.org/api/v1/json'
EXCLUDE_NAME = re.compile(r'qsar|dbworld|bioresponse|madelon|arcene|dexter|dorothea|gisette|'
                          r'cifar|mnist|fashion|text|news|reuters|spam|amazon|imdb|sentiment', re.I)


def get_json(url, retries=3):
  for attempt in range(retries):
    try:
      with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)
    except urllib.error.HTTPError as e:
      if e.code == 412:      # το OpenML το επιστρέφει όταν δεν υπάρχουν (άλλα) αποτελέσματα
        return None
      if attempt == retries - 1:
        raise
    except Exception:
      if attempt == retries - 1:
        raise
    time.sleep(2)


def list_all():
  rows, offset, limit = [], 0, 10000
  while True:
    data = get_json(f'{API}/data/list/status/active/limit/{limit}/offset/{offset}')
    if not data:
      break
    batch = data['data']['dataset']
    for d in batch:
      q = {i['name']: i['value'] for i in d.get('quality', [])}
      num = lambda k: float(q[k]) if q.get(k) not in (None, '') else float('nan')
      rows.append({'OpenML_ID': int(d['did']), 'Name': d['name'], 'Version': int(d.get('version', 1)),
                   'N': num('NumberOfInstances'), 'Columns': num('NumberOfFeatures'),
                   'Classes': num('NumberOfClasses'), 'Minority': num('MinorityClassSize'),
                   'Missing': num('NumberOfMissingValues')})
    if len(batch) < limit:
      break
    offset += limit
  return pd.DataFrame(rows)


def screen(df):
  p = df['Columns'] - 1
  used_ids = {i for c in DATASET_CONFIGS for i in c['identifiers'] if isinstance(i, int)}
  used_shapes = {(c['expected_n'], c['expected_p']) for c in DATASET_CONFIGS}
  out = df[(df['Classes'] == 2) & (p >= 1000) & (p > 5 * df['N']) & (df['N'] >= 40) &
           (df['Minority'] >= 10) & ~df['Name'].str.contains(EXCLUDE_NAME) &
           ~df['OpenML_ID'].isin(used_ids)].copy()
  out['p'] = (out['Columns'] - 1).astype(int)
  out['N'] = out['N'].astype(int)
  out['Possible_duplicate_of_used'] = [
      any(abs(n - en) <= 2 and abs(pp - ep) <= 2 for en, ep in used_shapes) for n, pp in zip(out['N'], out['p'])]
  # GEMLeR: AP_<ιστός>_<ιστός> / OVA_<ιστός> -> ιστοί, για έλεγχο κοινών δειγμάτων
  def tissues(name):
    m = re.match(r'^(AP|OVA)_(.+)$', name)
    return '+'.join(sorted(m.group(2).split('_'))) if m else ''
  out['GEMLeR_tissues'] = out['Name'].map(tissues)
  # Ίδιο όνομα σε πολλές εκδόσεις -> κρατάμε την πιο πρόσφατη
  out = out.sort_values('Version').groupby('Name', as_index=False).tail(1)
  return out.sort_values(['GEMLeR_tissues', 'Name']).reset_index(drop=True)


def details(did):
  d = get_json(f'{API}/data/{did}')['data_set_description']
  feats = get_json(f'{API}/data/features/{did}')
  target = d.get('default_target_attribute', '')
  labels = ''
  if feats:
    for f in feats['data_features']['feature']:
      if f['name'] == target:
        labels = ', '.join(f.get('nominal_value', [])) if isinstance(f.get('nominal_value'), list) \
            else str(f.get('nominal_value', ''))
  desc = re.sub(r'\s+', ' ', d.get('description', '') or '')
  return {'Target': target, 'Class_labels': labels, 'Citation': d.get('citation', ''),
          'Original_url': d.get('original_data_url', ''), 'Paper_url': d.get('paper_url', ''),
          'Description': desc[:1500]}


if __name__ == '__main__':
  print('Κατέβασμα λίστας datasets από το OpenML...', flush=True)
  allds = list_all()
  print(f'  {len(allds)} ενεργά datasets.')
  cand = screen(allds)
  print(f'  {len(cand)} περνούν τα αυτόματα κριτήρια.\n', flush=True)

  rows, lines = [], []
  for i, r in cand.iterrows():
    try:
      info = details(r['OpenML_ID'])
    except Exception as e:
      info = {'Target': '', 'Class_labels': f'ΣΦΑΛΜΑ: {e}', 'Citation': '', 'Original_url': '',
              'Paper_url': '', 'Description': ''}
    rows.append({**r.to_dict(), **{k: v for k, v in info.items() if k != 'Description'}})
    lines.append(f"{'=' * 90}\n[{r['OpenML_ID']}] {r['Name']}  (v{r['Version']})   N={r['N']}, p={r['p']}, "
                 f"μικρότερη κλάση={int(r['Minority'])}\n"
                 f"Κλάσεις ({info['Target']}): {info['Class_labels']}\n"
                 f"GEMLeR ιστοί: {r['GEMLeR_tissues'] or '-'}   Πιθανό διπλότυπο: {r['Possible_duplicate_of_used']}\n"
                 f"Citation: {info['Citation']}\nPaper: {info['Paper_url']}\nOriginal: {info['Original_url']}\n"
                 f"Περιγραφή: {info['Description']}\n")
    print(f"  [{r['OpenML_ID']:>6}] {r['Name'][:45]:<45} N={r['N']:<5} p={r['p']:<6} κλάσεις: {info['Class_labels'][:40]}",
          flush=True)

  pd.DataFrame(rows).to_csv('new_dataset_candidates.csv', index=False)
  with open('new_dataset_candidates_details.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
  print('\nΑποθηκεύτηκαν: new_dataset_candidates.csv, new_dataset_candidates_details.txt')
