# borrowed from https://github.com/bastings/nlp1-2017-projects/tree/master/dep-parser

import numpy as np
from collections import defaultdict
import networkx as nx

def mst(scores):
  """
  Chu-Liu-Edmonds' algorithm for finding minimum spanning arborescence in graphs.
  Calculates the arborescence with node 0 as root.
  Source: https://github.com/chantera/biaffineparser/blob/master/utils.py
  
  WARNING: mind the comment below. This mst function expects scores[i][j] to be the score from j to i, a
  not from i to j (as you would probably expect!). If you use a graph where you have the convention
  that a head points to its dependent, you will need to transpose it before calling this function.
  That is, call `mst(scores.T)` instead of `mst(scores)`.

  :param scores: `scores[i][j]` is the weight of edge from node `j` to node `i`
  :returns an array containing the head node (node with edge pointing to current node) for each node,
           with head[0] fixed as 0
  """
  length = scores.shape[0]
  scores = scores * (1 - np.eye(length))
  heads = np.argmax(scores, axis=1)
  heads[0] = 0
  tokens = np.arange(1, length)
  roots = np.where(heads[tokens] == 0)[0] + 1
  
  # print initial heads
  #print("initial heads:", heads)
  
  # deal with roots
  if len(roots) < 1:
    #print("no node is pointing to root, choosing one")
    root_scores = scores[tokens, 0]
    head_scores = scores[tokens, heads[tokens]]
    new_root = tokens[np.argmax(root_scores / head_scores)]
    #print("new root is:", new_root)
    heads[new_root] = 0
  elif len(roots) > 1:
    #print("multiple nodes are pointing to root, choosing one")
    root_scores = scores[roots, 0]
    scores[roots, 0] = 0
    new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
    new_root = roots[np.argmin(scores[roots, new_heads] / root_scores)]
    #print("new root is:", new_root)
    heads[roots] = new_heads
    heads[new_root] = 0

  # construct edges and vertices
  edges = defaultdict(set)
  vertices = set((0,))
  for dep, head in enumerate(heads[tokens]):
    vertices.add(dep + 1)
    edges[head].add(dep + 1)
      
  # identify cycles & contract
  for cycle in _find_cycle(vertices, edges):
    #print("Found cycle!", cycle)
    dependents = set()
    to_visit = set(cycle)
    while len(to_visit) > 0:
        node = to_visit.pop()
        #print("Contraction, visiting node:", node)
        if node not in dependents:
            dependents.add(node)
            to_visit.update(edges[node])
    cycle = np.array(list(cycle))
    old_heads = heads[cycle]
    old_scores = scores[cycle, old_heads]
    non_heads = np.array(list(dependents))
    scores[np.repeat(cycle, len(non_heads)),
           np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
    new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
    new_scores = scores[cycle, new_heads] / old_scores
    change = np.argmax(new_scores)
    changed_cycle = cycle[change]
    old_head = old_heads[change]
    new_head = new_heads[change]
    heads[changed_cycle] = new_head
    edges[new_head].add(changed_cycle)
    edges[old_head].remove(changed_cycle)

  return heads

def _find_cycle(vertices, edges):
  """
  Finds cycles in given graph, where the graph is provided as (vertices, edges).
  https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
  https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
  """
  _index = [0]
  _stack = []
  _indices = {}
  _lowlinks = {}
  _onstack = defaultdict(lambda: False)
  _SCCs = []

  def _strongconnect(v):
    _indices[v] = _index[0]
    _lowlinks[v] = _index[0]
    _index[0] += 1
    _stack.append(v)
    _onstack[v] = True

    for w in edges[v]:
      if w not in _indices:
        _strongconnect(w)
        _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
      elif _onstack[w]:
        _lowlinks[v] = min(_lowlinks[v], _indices[w])

    if _lowlinks[v] == _indices[v]:
      SCC = set()
      while True:
        w = _stack.pop()
        _onstack[w] = False
        SCC.add(w)
        if not (w != v):
          break
      _SCCs.append(SCC)

  for v in vertices:
    if v not in _indices:
      _strongconnect(v)

  return [SCC for SCC in _SCCs if len(SCC) > 1]
