# extract bert features for subject-verb agreement task 

import collections
import argparse
from tqdm import tqdm
import json
import csv
import random
random.seed(123)

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

def read_examples(linzen_input, train_prop, valid_prop, maxSeqLen):
  dependency_fields = ['sentence', 'orig_sentence', 'pos_sentence',
                     'subj', 'verb', 'subj_pos', 'has_rel', 'has_nsubj',
                     'verb_pos', 'subj_index', 'verb_index', 'n_intervening',
                     'last_intervening', 'n_diff_intervening', 'distance',
                     'max_depth', 'all_nouns', 'nouns_up_to_verb']
  # read all examples
  examples = []
  for i, d in enumerate(csv.DictReader(open(linzen_input), delimiter='\t')):
    examples.append({x: int(y) if y.isdigit() else y for x, y in d.items()})

  # create splits
  d = []
  for example in examples:
    tokens = example['orig_sentence'].split()
    if len(tokens) > maxSeqLen: continue
    d.append(example)
  random.shuffle(d)
  num_train = int(len(d)*train_prop)
  num_valid = int(len(d)*valid_prop)
  d_train, d_valid, d_test = d[0:num_train], d[num_train:num_train+num_valid], d[num_train+num_valid:]
  print('loaded instances = %d/%d/%d'%(len(d_train), len(d_valid), len(d_test)))
  return {'test':d_test, 'train':d_train, 'valid':d_valid}

def get_max_seq_length(instances, tokenizer):
  max_seq_len = -1
  for instance in instances:
    cand_tokens = tokenizer.tokenize(' '.join(instance['orig_sentence']))
    cur_len = len(cand_tokens)
    if cur_len > max_seq_len:
      max_seq_len = cur_len
  return max_seq_len

def convert_examples_to_features(instances, seqLen, tokenizer):
  features = []
  for (index, instance) in enumerate(instances):
    v = int(instance['verb_index']) - 1
    tokens = instance['orig_sentence'].split()
    cand_tokens = tokenizer.tokenize(' '.join(tokens[0:v])) + ['[MASK]'] +  tokenizer.tokenize(' '.join(tokens[(v+1):]))

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in cand_tokens:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seqLen:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seqLen
    assert len(input_mask) == seqLen
    assert len(input_type_ids) == seqLen

    features.append(
      InputFeatures(
        unique_id=index,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids))
  return features

def save(args, model, tokenizer, device):
  # convert data to ids
  examples = read_examples(args.data_file, 0.09, 0.01, 50) # default numbers obtained from Linzen et al.
  
  # extract and write features
  for s_name in examples:
    s_instances = examples[s_name] 
    output_file = args.output_folder + s_name + ".json"
    features = convert_examples_to_features(s_instances, seqLen=2+get_max_seq_length(s_instances, tokenizer), tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    pbar = tqdm(total=len(s_instances)//args.batch_size)
    with open(output_file, "w", encoding='utf-8') as writer:
      for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        for b, example_index in enumerate(example_indices):
          unique_id = example_index.item()
          output_json = collections.OrderedDict()
          output_json["linex_index"] = unique_id
          verb_index = s_instances[unique_id]['verb_index']-1
          layers = []
          for layer_index in range(len(all_encoder_layers)):
            layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
            layers.append([round(x.item(), 6) for x in layer_output[b][verb_index]])
          output_json["verb_layers"] = layers
          output_json["linzen_info"] = s_instances[unique_id]
          writer.write(json.dumps(output_json) + "\n")
        pbar.update(1)
    pbar.close()
    print('written features to %s'%output_file)

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

  parser.add_argument("--data_file",
                      default=None,
                      type=str,
                      required=True,
                      help="path to the data file for sva task from http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz")
  parser.add_argument("--output_folder",
                      default=None,
                      type=str,
                      required=True,
                      help="output folder where the features will be written")
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
                      default=8,
                      type=int,
                      help="total batch size for inference")
  parser.add_argument("--num_gpus",
                      default=1,
                      type=int,
                      help="no. of gpus to use")
  
  args = parser.parse_args()
  print(args)
  model, tokenizer, device = load(args)
  save(args, model, tokenizer, device)

if __name__ == "__main__":
  main()


