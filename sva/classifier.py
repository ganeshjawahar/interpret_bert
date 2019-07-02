# binary classifier for subject-verb agreement

import sys
import os
import argparse
import numpy as np
from senteval_tool import MLP
import json
import glob

def read_json(inp_file, layer):
  X, y = [], []
  with open(inp_file, 'r') as f:
    for line in f:
      json_info = json.loads(line.strip())        
      X.append(json_info['verb_layers'][layer])
      y.append(class_to_code[json_info['linzen_info']['verb_pos']])
  return np.array(X, dtype=np.float32), np.array(y)

def read_properties(inp_file):
  y_props = []
  with open(inp_file, 'r') as f:
    for line in f:
      json_info = json.loads(line.strip())
      y_props.append(json_info['linzen_info'])
  return y_props

class_to_code = {'VBZ': 0, 'VBP': 1}
def load(args):
  # extract the input features and labels
  train_X, train_y = read_json(args.input_folder+"train.json", args.layer)
  valid_X, valid_y = read_json(args.input_folder+"valid.json", args.layer)
  feat_dim = train_X.shape[1]
  print('dim=%d; #instances=%d/%d'%(feat_dim, train_X.shape[0], 
                                       valid_X.shape[0]))
  return train_X, train_y, valid_X, valid_y, feat_dim, len(class_to_code)

def classify(args, train_X, train_y, valid_X, valid_y, feat_dim, num_classes):
  classifier_config = {'nhid': args.nhid, 'optim': 'adam', 'batch_size': 64,
                  'tenacity': 5, 'epoch_size': 4, 'dropout': args.dropout}
  regs = [10**t for t in range(-5, -1)]
  props, scores = [], []
  for reg in regs:
    clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, 
              l2reg=reg, seed=args.seed, cudaEfficient=True)
    clf.fit(train_X, train_y, validation_data=(valid_X, valid_y))
    scores.append(round(100*clf.score(valid_X, valid_y), 2))
    props.append([reg])
  opt_prop = props[np.argmax(scores)]
  dev_acc = np.max(scores)
  clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, 
            l2reg=opt_prop[0], seed=args.seed, cudaEfficient=True)
  clf.fit(train_X, train_y, validation_data=(valid_X, valid_y))

  clf.model.eval()
  def gen_test_batch():
    reader = open(args.input_folder+"test.json", 'r')
    isDone = False
    while not isDone:
      X, y, props = [], [], []
      for bi in range(classifier_config['batch_size']):
        line = reader.readline()
        if not line:
          isDone = True
          break
        json_info = json.loads(line.strip())
        X.append(json_info['verb_layers'][args.layer])
        y.append(class_to_code[json_info['linzen_info']['verb_pos']])
        props.append(json_info['linzen_info'])
      if len(props) > 0:
        X = torch.FloatTensor(X).cuda()
        y = torch.LongTensor(y).cuda()
        yield [X, y, props]
    reader.close()
  overall_test_acc, num_test_records = 0, 0
  num_inst, score_inst = [0]*6, [0]*6
  import torch
  with torch.no_grad():
    for batch in gen_test_batch():
      X, y, props = batch
      output = clf.model(X)
      pred = output.data.max(1)[1].long().eq(y.data.long())
      overall_test_acc += pred.sum().item()
      num_test_records += len(props)
      for pi, prop in enumerate(props):
        cur_score = pred[pi].item()
        n_i  = prop['n_intervening']
        n_di = prop['n_diff_intervening']
        if n_i != n_di: continue
        if n_di in [0, 1, 2, 3, 4]:
          # specific case
          num_inst[n_di+1] += 1
          score_inst[n_di+1] += cur_score
          # all case
          num_inst[0] += 1
          score_inst[0] += cur_score
  overall_test_acc = round(100.0 * overall_test_acc / num_test_records, 2)
  res = 'overall/0/1/2/3/4 = '
  for i in range(len(num_inst)):
    if num_inst[i]==0:
      res += '0/'
    else:
      res += str(round(100*score_inst[i]/num_inst[i], 2)) + "/"
  print(res)

def main():
  parser = argparse.ArgumentParser(description="Subject-verb agreement using \
                                 simple ML model")
  parser.add_argument("--input_folder", type=str, default=None, 
                      help="folder containing bert features and labels")
  parser.add_argument('--layer', type=int, default=0, help='bert layer to be \
                      probed?')
  parser.add_argument('--nhid', type=int, default=50, help='hidden size of MLP')
  parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
  parser.add_argument('--seed', type=int, default=123, help='seed value to be \
                      set manually')
  args = parser.parse_args()
  print(args)
  train_X, train_y, valid_X, valid_y, feat_dim, num_classes = load(args)
  classify(args, train_X, train_y, valid_X, valid_y, feat_dim, num_classes)

if __name__ == "__main__":
  main()


