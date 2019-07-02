# Tensor Product Decomposition Network to approximate BERT

import sys
import json
import io
import os
from tqdm import tqdm
import argparse
import random
import numpy as np
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from treerole_helper import gen_treerole, gen_rand_tree

def set_seed(args):
  use_cuda = torch.cuda.is_available()
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  if use_cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
  device = torch.device("cuda" if use_cuda and 
                        not args.no_cuda else "cpu")
  return device

def load_bert(args):
  # load bert tokenizer and model
  tokenizer = BertTokenizer.from_pretrained(args.bert_model, 
              do_lower_case=True,
              cache_dir=args.cache_dir)
  pretrained_model = BertModel.from_pretrained(args.bert_model, 
              cache_dir=args.cache_dir)
  return tokenizer, pretrained_model

# role scheme generator
def get_roles(args, tokens, parse_info=None, unmasked_tokens=None):
  if args.role_scheme == 'l2r':
    return [i for i in range(len(tokens))]
  if args.role_scheme == 'r2l':
    return [len(tokens)-i-1 for i in range(len(tokens))]
  if args.role_scheme == 'bow':
    return [0] *len(tokens)
  if args.role_scheme == 'bidi':
    return ['%d-%d'%(i, len(tokens)-i-1) for i in range(len(tokens))]
  if args.role_scheme == 'tree':
    tokens = unmasked_tokens if unmasked_tokens else tokens
    tree_roles = None
    if not args.rand_tree:
      tree_roles = gen_treerole(tokens[1:-1], parse_info)
    else:
      tree_roles = gen_rand_tree(len(tokens)-2)
    return ['[CLS]'] + tree_roles + ['[SEP]']

# create word dictionaries
def create_vocab(args, tokenizer):
  id2token, token2id = {0:'[UNK]'}, {'[UNK]':0}
  for file in ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', \
                                          'snli_1.0_test.jsonl']:
    with open(args.input_folder + "/bert_" + file, 'r') as f:
      li = 0
      for line in f:
        content = json.loads(line.strip())
        premise = content['sentence']
        tokens = ['[CLS]'] + tokenizer.tokenize(premise) + \
              ['[SEP]'] if 'tokens' not in content else content['tokens']
        unmasked_tokens = ['[CLS]'] + tokenizer.tokenize(premise) + \
              ['[SEP]'] if '[MASK]' in tokens else None
        for token in tokens:
          if token not in token2id:
            token2id[token] = len(id2token)
            id2token[token2id[token]] = token
        li += 1
        if li == 1000:
          break
  print('filler (token) count = %d'%len(id2token))
  return id2token, token2id

# load pretrained embeddings
def load_pretrained_embed(args, pretrained_model, tokenizer, id2token):
  if not args.pretrained_embedding:
    return None
  weights = pretrained_model.embeddings.word_embeddings.weight.detach().numpy()
  pretrained_embeddings = np.random.rand(len(id2token), weights.shape[1])
  bert_token_ids = tokenizer.convert_tokens_to_ids([id2token[ti] for ti in range(len(id2token))])
  assert(len(bert_token_ids) == len(id2token))
  for i, bti in enumerate(bert_token_ids):
    pretrained_embeddings[i] = weights[bti]
  print('loaded pretrained embeddings')
  return pretrained_embeddings

# create model inputs
class Instance(object):
  def __init__(self, filler, role, rep):
    self.filler = filler
    self.role = role
    self.rep = rep
def get_representation(content, args):
  if args.layer == -1:
    return content['pooled_output']
  return content['features'][0]['layers'][args.layer]['values']
id2role, role2id = {0:'[UNK]'}, {'[UNK]':0}
def read_as_tensors(args, inp_file, tokenizer, token2id, device):
  data = []
  with open(inp_file, 'r') as f:
    li = 0
    for line in f:
      content = json.loads(line.strip())
      # create X_filler
      premise = content['sentence']
      tokens = ['[CLS]'] + tokenizer.tokenize(premise) + \
            ['[SEP]'] if 'tokens' not in content else content['tokens']
      unmasked_tokens = ['[CLS]'] + tokenizer.tokenize(premise) + \
            ['[SEP]'] if '[MASK]' in tokens else None
      filler_t = Variable(torch.LongTensor([token2id[token] for token in tokens]), 
                          requires_grad=False).unsqueeze(0)
      filler_t = filler_t.to(device)
      # create X_role
      roles = get_roles(args, tokens, parse_info=content['binaryParse'] 
                        if 'binaryParse' in content else None,
                        unmasked_tokens=unmasked_tokens)
      assert(len(roles) == len(tokens))
      for role in roles:
        if role not in role2id:
          role2id[role] = len(id2role)
          id2role[role2id[role]] = role
      role_t = Variable(torch.LongTensor([role2id[role] for role in roles]), 
                        requires_grad=False).unsqueeze(0)
      role_t = role_t.to(device)
      # create Y_target
      representation = get_representation(content, args)
      rep_t = Variable(torch.FloatTensor(representation), 
                       requires_grad=False).unsqueeze(0)
      rep_t = rep_t.to(device)
      data.append(Instance(filler_t, role_t, rep_t))
      li = li + 1
      if li == 1000:
        break
  print('read %d instances from %s'%(len(data), inp_file))
  return data

def read_tensors(args, tokenizer, token2id, device):
  train_data = read_as_tensors(args, args.input_folder + "/bert_snli_1.0_train.jsonl", tokenizer, token2id, device)
  valid_data = read_as_tensors(args, args.input_folder + "/bert_snli_1.0_dev.jsonl", tokenizer, token2id, device)
  test_data = read_as_tensors(args, args.input_folder + "/bert_snli_1.0_test.jsonl", tokenizer, token2id, device)
  print('role count = %d'%len(id2role))
  return train_data, valid_data, test_data

# Model definition
# Defines the tensor product, used in tensor product representations
class SumFlattenedOuterProduct(nn.Module):
  def __init__(self):
    super(SumFlattenedOuterProduct, self).__init__()

  def forward(self, input1, input2):    
    # This layer will take the sum flattened outer product of the filler
    # and role embeddings
    #outer_product = torch.mm(input1.t(), input2)
    #flattened_outer_product = outer_product.view(-1).unsqueeze(0)
    outer_product = torch.bmm(input1.transpose(1,2), input2)
    flattened_outer_product = outer_product.view(outer_product.size()[0],-1).unsqueeze(0)
    sum_flattened_outer_product = flattened_outer_product
    return sum_flattened_outer_product

# A tensor product encoder layer 
# Takes a list of fillers and a list of roles and returns an encoding
class TensorProductEncoder(nn.Module):
  def __init__(self, num_roles, num_fillers, role_dim, filler_dim, 
               final_layer_width, pretrained_embeddings, untrained):
    super(TensorProductEncoder, self).__init__()
    self.role_embed = nn.Embedding(num_roles, role_dim)
    self.filler_embed = None
    if pretrained_embeddings is None:
      self.filler_embed = nn.Embedding(num_fillers, filler_dim)
    else:
      filler_dim = pretrained_embeddings.shape[1]
      pretrained_embed = nn.Embedding(num_fillers, filler_dim)
      if not untrained:
        pretrained_embed.weight.data = torch.from_numpy(pretrained_embeddings).float()
      pretrained_embed.weight.requires_grad = False
      self.filler_embed = nn.Sequential(pretrained_embed, 
                          nn.Linear(filler_dim, filler_dim)) if not untrained else pretrained_embed
    if untrained:
      self.filler_embed.weight.requires_grad = False
      self.role_embed.weight.requires_grad = False
    self.sum_layer = SumFlattenedOuterProduct()
    self.last_layer = nn.Linear(filler_dim * role_dim, final_layer_width)

  def forward(self, fillers, roles):
    filler_embed = self.filler_embed(fillers)
    role_embed = self.role_embed(roles)
    output = self.sum_layer(filler_embed, role_embed)
    output = self.last_layer(output)
    return output

# batchify the data
def batchify(data, batch_size):
  len2inst = {}
  for di, datum in enumerate(data):
    dlen = datum.filler.size(1)
    if dlen not in len2inst:
      len2inst[dlen] = []
    len2inst[dlen].append(di)
  batches = []
  for dlen in sorted(len2inst.keys()):
    ids = len2inst[dlen]
    for bi in range(len(ids)//batch_size):
      first_item = data[ids[bi*batch_size]]
      cur_filler, cur_role, cur_rep = first_item.filler, first_item.role, \
                                      first_item.rep
      for idi in ids[(bi*batch_size)+1: (bi+1)*batch_size]:
        inst = data[idi]
        cur_filler = torch.cat((cur_filler, inst.filler), 0)
        cur_role = torch.cat((cur_role, inst.role), 0)
        cur_rep = torch.cat((cur_rep, inst.rep), 0)
      batches.append((cur_filler, cur_role, cur_rep.unsqueeze(0)))
  return batches

def run_tpdn(args, id2token, token2id, pretrained_embeddings, train_data, valid_data, test_data, device):
  # create the model
  model = TensorProductEncoder(len(role2id), len(token2id), args.role_dim, 
            args.filler_dim, train_data[0].rep.size(1), pretrained_embeddings,
            args.untrained)
  model.to(device)

  # training configuration (adapted from McCoy et al.)
  batch_size = 32
  learning_rate = 0.001
  n_epochs = 100
  patience = 10
  best_loss = 1000000
  training_done = 0
  count_not_improved = 0
  print_every = 1000//32
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.MSELoss()
  
  # batchify train
  train_batches = batchify(train_data, batch_size)

  def train(batch):
    # zero the gradient
    optimizer.zero_grad() 
    # iterate over this batch
    cur_loss = 0
    for items in batch:
      filler_in, role_in, target_out = items
      enc_out = model(filler_in, role_in)
      cur_loss += criterion(enc_out, target_out)
    cur_loss.backward()
    optimizer.step()
    enc_out = enc_out.detach()
    return cur_loss.data.item()

  def eval(data):
    total_mse = 0.
    for instance in data:
      total_mse += torch.mean(torch.pow(model(instance.filler, 
                instance.role).data - instance.rep.data, 2))
    return total_mse/len(data)

  # start the training
  for epoch in range(n_epochs):
    improved_this_epoch = 0
    random.shuffle(train_batches)
    for bi in tqdm(range(len(train_batches)//batch_size)):
      loss = train(train_batches[bi*batch_size: (bi+1)*batch_size])
      if bi%print_every == 0:
        valid_mse = eval(valid_data)
        if valid_mse < best_loss:
          best_loss = valid_mse
          improved_this_epoch = 1
          count_not_improved = 0
          torch.save(model.state_dict(), args.output_folder + "/" + args.run_name)
        else:
          count_not_improved += 1
          if count_not_improved == patience:
            training_done = 1
            break
    if training_done:
      break
    if not improved_this_epoch:
      break
    print('epoch %d done. best valid. loss = %.4f'%(epoch+1, best_loss))

  # MSE evaluation
  model.load_state_dict(torch.load(args.output_folder + "/" + args.run_name))
  print('valid loss = %.4f, test loss = %.4f'%(best_loss, eval(test_data)))
  os.remove(args.output_folder + "/" + args.run_name)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_folder", help="folder containing BERT \
                      representations", type=str, default=None)
  parser.add_argument("--output_folder", help="folder to store tpdn intermediate \
                      models", type=str, default=None)
  parser.add_argument("--layer", help="BERT layer (indexed from 0) to be \
                      approximated. -1 for pooled_output", type=int, default=-1)
  parser.add_argument("--role_scheme", help="role scheme to use. l2r, r2l, bow, \
                      bidi, tree", type=str, default='l2r')
  parser.add_argument("--rand_tree", help="substitute randomized binary tree.\
                      used when role_scheme=tree", action='store_true')
  parser.add_argument("--filler_dim", help="embedding dimension for fillers", 
                      type=int, default=10)
  parser.add_argument("--role_dim", help="embedding dimension for roles", 
                      type=int, default=6)
  parser.add_argument("--pretrained_embedding", help="use pretrained \
                      word embeddings?", action='store_false')
  parser.add_argument("--bert_model", help="variant of BERT model used to \
                      generate representations", type=str, default="bert-base-uncased")
  parser.add_argument("--seed", help="seed value for pseudo random generator", 
                      type=int, default=123)
  parser.add_argument("--no_cuda", help="use only CPU", action='store_true')
  parser.add_argument("--run_name", help="name for current run", type=str, 
                      default="demo_run")
  parser.add_argument("--untrained", help="don't update TPDN parameters except final layer",                  action='store_true')
  parser.add_argument("--cache_dir", default='/tmp', type=str,
                      help="directory to cache bert pre-trained models")
  args = parser.parse_args()
  print(args)
  device = set_seed(args)
  tokenizer, pretrained_model = load_bert(args)
  id2token, token2id = create_vocab(args, tokenizer)
  pretrained_embeddings = load_pretrained_embed(args, pretrained_model, tokenizer, id2token)
  train_data, valid_data, test_data = read_tensors(args, tokenizer, token2id, device)
  run_tpdn(args, id2token, token2id, pretrained_embeddings, train_data, valid_data, test_data, device)

if __name__ == "__main__":
  main()


