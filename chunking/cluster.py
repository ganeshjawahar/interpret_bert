# cluster chunking features

import argparse
import json

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
np.random.seed(123)

def cluster(args):
  idx = np.random.permutation(3500) 
  train_idx = idx[0:3000]
  test_idx = idx[3000:]

  num_layers, layer_id = None, 0
  while True:
    # extract X, labels <= create tsne input
    X, y = [], []
    id2lab, lab2id = {}, {}
    with open(args.feat_file, 'r') as f:
      for line in f:
        info = json.loads(line.strip())
        span_start = np.array(info['start_layer'][layer_id], dtype=np.float32)
        span_end = np.array(info['end_layer'][layer_id], dtype=np.float32)
        label = info['label']
        if label not in lab2id:
          lab2id[label] = len(lab2id)
          id2lab[lab2id[label]] = label
        y.append(lab2id[label])
        X.append(np.concatenate((span_start, span_end, np.multiply(span_start, span_end), span_start-span_end)))
        if not num_layers:
          num_layers = len(info['end_layer'])

    train_X, train_y, test_X, test_y = [], [], [], []
    for idi in train_idx:
      train_X.append(X[idi])
      train_y.append(y[idi])
    for idi in test_idx:
      test_X.append(X[idi])
      test_y.append(y[idi])

    kmeans = KMeans(n_clusters=len(lab2id), random_state=123).fit(train_X)
    pred_y = kmeans.predict(test_X)
    layer_id += 1
    print('layer %d => NMI = %.2f'%(layer_id, normalized_mutual_info_score(test_y, pred_y)))
    
    if layer_id == num_layers:
      break

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--feat_file",
                      default=None,
                      type=str,
                      required=True,
                      help="file containing the features")
  
  args = parser.parse_args()
  print(args)
  cluster(args)

if __name__ == "__main__":
  main()

