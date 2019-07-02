# visualize chunking features

import argparse
import numpy as np
import json

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter

def visualize(args):
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
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int)
    layer_id += 1

    # perform t-SNE
    embeddings = TSNE(n_components=2, init='pca', verbose=0, perplexity=30, n_iter=500).fit_transform(X)
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_classes = len(lab2id)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))
    labels = np.arange(num_classes)
    for i in range(num_classes):
      ax.scatter(xx[y==i], yy[y==i], color=colors[i], label=id2lab[i], s=12)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(args.output_file_prefix + str(layer_id) +".pdf", format='pdf', dpi=600)
    #plt.show()
    print('layer %d plot => %s'%(layer_id, args.output_file_prefix + str(layer_id) +".pdf"))
    
    if layer_id == num_layers:
      break

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--feat_file",
                      default=None,
                      type=str,
                      required=True,
                      help="file containing the features")
  parser.add_argument("--output_file_prefix",
                      default=None,
                      type=str,
                      required=True,
                      help="prefix of output file where the graphs will be written")

  args = parser.parse_args()
  print(args)
  visualize(args)

if __name__ == "__main__":
  main()

