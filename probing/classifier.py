# train and evaluate classifier for each probing task

import argparse
import json

import numpy as np
from senteval_tool import MLP

def load(args):
  reader_l = open(args.labels_file, 'r', encoding='utf-8')
  reader_f = open(args.feats_file, 'r')
  inst, feat_dim, cat2id, id2cat = 0, -1, {}, {}
  train_X, train_y, dev_X, dev_y, test_X, test_y = [], [], [], [], [], []
  while True:
    label_l = reader_l.readline()
    if not label_l:
      break
    feat_l = reader_f.readline()
    feat_json = json.loads(feat_l)
    assert(feat_json['linex_index']==inst)
    assert(len(feat_json['features'])==1)
    X = None
    for layer in feat_json['features'][0]['layers']:
      if layer['index']==args.layer:
        X = layer['values']
        break
    assert(X is not None)
    if feat_dim < 0:
      feat_dim = len(X)
    split, lab, text = label_l.split('\t')
    if lab not in cat2id:
      cat2id[lab] = len(id2cat)
      id2cat[cat2id[lab]] = lab
    y = cat2id[lab]
    if split == 'tr':
      train_X.append(X)
      train_y.append(y)
    elif split == 'va':
      dev_X.append(X)
      dev_y.append(y)
    elif split == 'te':
      test_X.append(X)
      test_y.append(y)
    inst += 1
  reader_l.close()
  reader_f.close()
  train_X = np.array(train_X, dtype=np.float32)
  dev_X = np.array(dev_X, dtype=np.float32)
  test_X = np.array(test_X, dtype=np.float32)
  train_y = np.array(train_y)
  dev_y = np.array(dev_y)
  test_y = np.array(test_y)
  print('loaded %d/%d/%d samples; %d labels;'%(train_X.shape[0], dev_X.shape[0], test_X.shape[0], len(cat2id)))
  return train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, len(cat2id)

def classify(args, train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes):
  classifier_config = {'nhid': args.nhid, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4, 'dropout': args.dropout}
  regs = [10**t for t in range(-5, -1)]
  props, scores = [], []
  for reg in regs:
    clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=reg, seed=args.seed, cudaEfficient=True)
    clf.fit(train_X, train_y, validation_data=(dev_X, dev_y))
    scores.append(round(100*clf.score(dev_X, dev_y), 2))
    props.append([reg])
  opt_prop = props[np.argmax(scores)]
  dev_acc = np.max(scores)
  clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=opt_prop[0], seed=args.seed, cudaEfficient=True)
  clf.fit(train_X, train_y, validation_data=(dev_X, dev_y))
  test_acc = round(100*clf.score(test_X, test_y), 2)
  print("best reg = %.2f; dev score = %.4f; test score = %.4f;"%(opt_prop[0], dev_acc, test_acc))

def main():
  parser = argparse.ArgumentParser(description="Probing classifier")
  parser.add_argument("--labels_file", 
                      type=str, 
                      default=None, 
                      help="file containing probing text and labels")
  parser.add_argument("--feats_file", 
                      type=str,
                      default=None, 
                      help="file containing bert features for a probing task")
  parser.add_argument('--layer', 
                      type=int, 
                      default=0, 
                      help='bert layer id to probe')
  parser.add_argument('--nhid', 
                      type=int, 
                      default=50, 
                      help='hidden size of MLP')
  parser.add_argument('--dropout', 
                      type=float, 
                      default=0.0, 
                      help='dropout prob. value')
  parser.add_argument('--seed', 
                      type=int, 
                      default=123, 
                      help='seed value to be set manually')

  args = parser.parse_args()
  print(args)
  train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes = load(args)
  classify(args, train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes)

if __name__ == "__main__":
  main()
