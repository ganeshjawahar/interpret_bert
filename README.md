## What does BERT learn about the structure of language?

Code used in our [ACL'19 paper](https://drive.google.com/open?id=166ngGwApN5XdOnUzs_y12GqdDCoPvUeh) for interpreting [BERT model](https://arxiv.org/abs/1810.04805).

### Dependencies
* [PyTorch](https://pytorch.org/)
* [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
* [SentEval](https://github.com/facebookresearch/SentEval)
* [spaCy](https://spacy.io/) (for dependency tree visualization)

### Quick Start

#### Phrasal Syntax (Section 3 in paper)
* Navigate:
```
cd chunking/
```
* Download the train set from [CoNLL-2000 chunking corpus](https://www.clips.uantwerpen.be/conll2000/chunking/):
```
wget https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
gunzip train.txt.gz
```
The last command replaces `train.txt.gz` file with `train.txt` file.
* Extract BERT features for chunking related tasks (clustering and visualization):
```
python extract_features.py --train_file train.txt --output_file chunking_rep.json
```
* Run t-SNE of span embeddings for each BERT layer (Figure 1):
```
python visualize.py --feat_file chunking_rep.json --output_file_prefix tsne_layer_
```
This would create one t-SNE plot for each BERT layer and stores as pdf (e.g. `tsne_layer_0.pdf`).
* Run KMeans to evaluate the clustering performance of span embeddings for each BERT layer (Table 1):
```
python cluster.py --feat_file chunking_rep.json
```

#### Probing Tasks (Section 4)
* Navigate:
```
cd probing/
```
* Download the [data files for 10 probing tasks](https://github.com/facebookresearch/SentEval/tree/master/data/probing) (e.g. `tree_depth.txt`)
* Extract BERT features for sentence level probing tasks (`tree_depth` in this case):
```
python extract_features.py --data_file tree_depth.txt --output_file tree_depth_rep.json
```
In the above command, append `--untrained_bert` flag to extract untrained BERT features.
* Train the probing classifier for a given BERT layer (indexed from 0) and evaluate the performance (Table 2):
```
python classifier.py --labels_file tree_depth.txt --feats_file tree_depth_rep.json --layer 0
```
We use the hyperparameter search space recommended by [SentEval](https://arxiv.org/abs/1803.05449).

#### Subject-Verb Agreement (SVA) (Section 5)
* Navigate:
```
cd sva/
```
* Download the [data file for SVA task](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz) and extract it.
* Extract BERT features for SVA task:
```
python extract_features.py --data_file agr_50_mostcommon_10K.tsv --output_folder ./
``` 
* Train the binary classifier for a given BERT layer (indexed from 0) and evaluate the performance (Table 3):
```
python classifier.py --input_folder ./ --layer 0
```
We use the hyperparameter search space recommended by [SentEval](https://arxiv.org/abs/1803.05449).

#### Compositional Structure (Section 6)
* Navigate:
```
cd tpdn/
```
* Download the [SNLI 1.0 corpus](https://nlp.stanford.edu/projects/snli/) and extract it.
* Extract BERT features for premise sentences present in SNLI:
```
python extract_features.py --input_folder . --output_folder .
```
* Train the Tensor Product Decomposition Network (TPDN) to approximate a given BERT layer (indexed from 0) and evaluate the performance (Table 4):
```
python approx.py --input_folder . --output_folder . --layer 0
```
Check `--role_scheme` and `--rand_tree` flags for setting the role scheme.
* Induce dependency parse tree from attention weights for a given attention head and BERT layer (both indexed from 1) (Figure 2):
```
python induce_dep_trees.py --sentence text "The keys to the cabinet are on the table" --head_id 11 --layer_id 2 --sentence_root 6 
```

### Acknowledgements
This repository would not be possible without the efforts of the creators/maintainers of the following libraries:
* [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) from huggingface
* [SentEval](https://github.com/facebookresearch/SentEval) from facebookresearch
* [bert-syntax](https://github.com/yoavg/bert-syntax) from yoavg
* [tpdn](https://github.com/tommccoy1/tpdn) from tommccoy1
* [rnn_agreement](https://github.com/TalLinzen/rnn_agreement) from TalLinzen
* [Chu-Liu-Edmonds](https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/mst/mst.ipynb) from bastings

### License
This repository is GPL-licensed.

