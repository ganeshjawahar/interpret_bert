# extract bert features from SNLI corpus (for TPDN training)

import os
import codecs
import collections
import argparse
from tqdm import tqdm
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

class InputExample(object):
  def __init__(self, unique_id, text, parse, binaryParse):
    self.unique_id = unique_id
    self.text = text
    self.parse = parse
    self.binaryParse = binaryParse

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

def read_examples(input_file, sample_cache):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with codecs.open(input_file, "r", encoding='utf-8') as reader:
    for line in reader:
      content = json.loads(line.strip())
      if content['sentence1'] in sample_cache:
        continue
      sample_cache[content['sentence1']] = True
      examples.append(InputExample(unique_id=unique_id, 
                                   text=content['sentence1'],
                                   parse=content['sentence1_parse'],
                                   binaryParse=content['sentence1_binary_parse']))
      unique_id += 1
  return examples, sample_cache

def get_max_seq_length(examples, tokenizer):
  max_seq_len = -1
  for example in examples:
    cand_tokens = tokenizer.tokenize(example.text)
    cur_len = len(cand_tokens)
    if cur_len > max_seq_len:
      max_seq_len = cur_len
  return max_seq_len

def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""
  features = []
  for (ex_index, example) in enumerate(examples):
    cand_tokens = tokenizer.tokenize(example.text)
    # Account for [CLS] and [SEP] with "- 2"
    if len(cand_tokens) > seq_length - 2:
      cand_tokens = cand_tokens[0:(seq_length - 2)]

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
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    features.append(
      InputFeatures(
        unique_id=example.unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids))
  return features

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

def save(args, model, tokenizer, device):
  sample_cache = {}
  for file in ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', \
                                          'snli_1.0_test.jsonl']:
    assert(os.path.exists(args.input_folder + "/" + file))
    input_file = args.input_folder + "/" + file
    output_file = args.output_folder + "/bert_" + file

    # extract ids for instances
    examples, sample_cache = read_examples(input_file, sample_cache)
    print('%d samples for %s task'%(len(examples), file))
    features = convert_examples_to_features(
          examples=examples, seq_length=2+get_max_seq_length(examples, 
                                                             tokenizer),
          tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], 
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], 
                                  dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, 
                                 batch_size=args.batch_size)

    pbar = tqdm(total=len(examples)//args.batch_size)
    with open(output_file, "w", encoding='utf-8') as writer:
      for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_encoder_layers, pooled_layer = model(input_ids, token_type_ids=None,
                                                       attention_mask=input_mask)
        pooled_layer = pooled_layer.detach().cpu().numpy()
        for b, example_index in enumerate(example_indices):
          feature = features[example_index.item()]
          unique_id = int(feature.unique_id)
          output_json = collections.OrderedDict()
          output_json["linex_index"] = unique_id
          output_json["sentence"] = examples[unique_id].text
          output_json["parse"] = examples[unique_id].parse
          output_json["binaryParse"] = examples[unique_id].binaryParse
          output_json["pooled_output"] = [
            round(x.item(), 6) for x in pooled_layer[b]
          ]
          all_out_features = []
          for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for layer_index in range(len(all_encoder_layers)):
              layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
              layer_output = layer_output[b]
              layers = collections.OrderedDict()
              layers["index"] = layer_index
              layers["values"] = [
                  round(x.item(), 6) for x in layer_output[i]
              ]
              all_layers.append(layers)
            out_features = collections.OrderedDict()
            out_features["token"] = token
            out_features["layers"] = all_layers
            all_out_features.append(out_features)
            break
          output_json["features"] = all_out_features
          writer.write(json.dumps(output_json) + "\n")
        pbar.update(1)
    pbar.close()
    print('written features to %s'%output_file)

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--input_folder",
                      default=None,
                      type=str,
                      required=True,
                      help="path to the SNLI dataset from https://nlp.stanford.edu/projects/snli/")
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





