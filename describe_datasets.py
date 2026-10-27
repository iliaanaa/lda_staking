"""
describe_datasets.py — Κατεβάζει την περιγραφή (description) κάθε dataset από το OpenML,
ώστε να δούμε τι σημαίνει κάθε ετικέτα κλάσης (π.χ. αν '1' = καρκίνος ή υγιές).
Έξοδος: datasets_descriptions.txt
"""
import numpy as np
from sklearn.datasets import fetch_openml

from common import DATASET_CONFIGS

if __name__ == '__main__':
  with open('datasets_descriptions.txt', 'w', encoding='utf-8') as f:
    for cfg in DATASET_CONFIGS:
      data = fetch_openml(data_id=cfg['identifiers'][0], as_frame=True, parser='auto')
      labels, counts = np.unique(np.asarray(data.target).astype(str), return_counts=True)
      d = data.details
      block = (f"{'=' * 80}\n{cfg['display_name']}  (OpenML id={d.get('id')}, name={d.get('name')})\n"
               f"Κλάσεις: {dict(zip(labels, counts.tolist()))}\n"
               f"Target column: {d.get('default_target_attribute')}\n"
               f"Original data url: {d.get('original_data_url', '-')}\n"
               f"Citation: {d.get('citation', '-')}\n"
               f"Paper url: {d.get('paper_url', '-')}\n\n"
               f"DESCRIPTION:\n{data.DESCR}\n\n")
      print(block)
      f.write(block)
  print('Αποθηκεύτηκε: datasets_descriptions.txt')
