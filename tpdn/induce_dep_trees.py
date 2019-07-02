# induce trees from attention mechanisms based on 
# Section 6 of http://aclweb.org/anthology/W18-5431

import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from unsuptree_modeler import BertModel
from mst import mst

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

def get_max_seq_length(examples, tokenizer):
  max_seq_len = -1
  for example in examples:
    cand_tokens = tokenizer.tokenize(example)
    cur_len = len(cand_tokens)
    if cur_len > max_seq_len:
      max_seq_len = cur_len
  return max_seq_len

def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""
  features = []
  for (ex_index, example) in enumerate(examples):
    cand_tokens = tokenizer.tokenize(example)
    # Account for [CLS] and [SEP] with "- 2"
    if len(cand_tokens) > seq_length - 2:
      cand_tokens = cand_tokens[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for ti, token in enumerate(cand_tokens):
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    features.append(
      InputFeatures(
        unique_id=ex_index,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids))
  return features

def save(args, model, tokenizer, device):
  # convert data to ids
  examples = [args.sentence_text]
  features = convert_examples_to_features(
        examples=examples, seq_length=2+get_max_seq_length(examples, 
                                                           tokenizer),
        tokenizer=tokenizer)

  # extract and write dependency parses
  all_input_ids = torch.tensor([f.input_ids for f in features], 
                               dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], 
                              dtype=torch.long)
  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, 
                             batch_size=args.batch_size)
  for input_ids, input_mask, example_indices in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    all_encoder_layers, pooled_layer, raw_attn_layers = model(input_ids, 
                                                      token_type_ids=None,
                                                 attention_mask=input_mask)
    cur_tokens = features[example_indices[0]].tokens[1:-1]
    cur_layer = raw_attn_layers[args.layer_id-1].squeeze()
    cur_head = cur_layer[args.head_id-1]
    cur_attn_matrix = cur_head[0:len(cur_tokens)+1, 0:len(cur_tokens)+1].detach().cpu().numpy()
    cur_attn_matrix[:,0] = -1.
    cur_attn_matrix[args.sentence_root,0] = 1.0
    np.fill_diagonal(cur_attn_matrix, -1.)
    mst_out = mst(cur_attn_matrix)
    tokens = ['<root>'] + cur_tokens
    print('tokens ==>')
    print(tokens)
    print('heads ==>')
    print([tokens[head_id] for head_id in mst_out])
    break

def load(args):
  print('loading %s model'%args.bert_model)
  device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, cache_dir=args.cache_dir)
  model = BertModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
  model.to(device)
  if args.num_gpus > 1:
    model = torch.nn.DataParallel(model)
  model.eval()
  return model, tokenizer, device

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sentence_text",
                      default="The keys to the cabinet are on the table",
                      type=str,
                      required=True,
                      help="sentence for which we need to induce dependency trees")
  parser.add_argument("--cache_dir",
                      default='/tmp',
                      type=str,
                      help="directory to cache bert pre-trained models")
  parser.add_argument("--bert_model", 
                      default="bert-base-uncased", 
                      type=str,
                      help="bert pre-trained model selected in the list: bert-base-uncased, "
                      "bert-large-uncased, bert-base-cased, bert-large-cased")
  parser.add_argument("--no_cuda",
                      action='store_true',
                      help="whether not to use CUDA when available")
  parser.add_argument("--batch_size",
                      default=1,
                      type=int,
                      help="total batch size for inference")
  parser.add_argument("--num_gpus",
                      default=1,
                      type=int,
                      help="no. of gpus to use")
  parser.add_argument("--head_id",
                      default=11,
                      type=int,
                      required=True,
                      help="head identifier in multi-headed attention?")
  parser.add_argument("--layer_id",
                      default=2,
                      type=int,
                      required=True,
                      help="layer identifier for multi-layered transformer")
  parser.add_argument("--sentence_root",
                      default=6,
                      type=int,
                      required=True,
                      help="identifier of root token in sentence")
  
  args = parser.parse_args()
  print(args)
  model, tokenizer, device = load(args)
  save(args, model, tokenizer, device)

if __name__ == "__main__":
  main()


