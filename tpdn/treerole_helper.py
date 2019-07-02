# utilities to create roles

class Node:
  def __init__(self, data, idi):
    self.data = data
    self.idi = idi
    self.left, self.right = None, None

def simple_case(tokens, ids):
  assert(tokens[0]=='(')
  assert(tokens[-1]==')')
  root = Node('latent', -1)
  root.left = Node(tokens[1], ids[1])
  root.right = Node(tokens[2], ids[2])
  return root

def choose_split_index(tokens, ids):
  tokens = tokens[1:-1]
  ids = ids[1:-1]
  if tokens[0] != '(':
    return [tokens[0]], tokens[1:], [ids[0]], ids[1:] 
  num_left, i = 0, 0
  while i < len(tokens)-1:
    if tokens[i] == '(':
      num_left += 1
    if tokens[i] == ')':
      num_left -= 1
      if num_left == 0:
        break
    i = i + 1
  return tokens[0:i+1], tokens[i+1:], ids[0:i+1], ids[i+1:]

def create_btree(tokens, ids):
  if len(tokens) == 0:
    return None
  if len(tokens) == 1:
    return Node(tokens[0], ids[0])
  if len(tokens) == 4:
    return simple_case(tokens, ids)
  assert len(tokens)>4
  left_tokens, right_tokens, left_ids, right_ids = choose_split_index(tokens, ids)
  root = Node('latent', -1)
  root.left = create_btree(left_tokens, left_ids)
  root.right = create_btree(right_tokens, right_ids)
  return root

def create_roles(tree, roles, cur_path):
  if not tree:
    return
  if not tree.left and not tree.right:
    roles[tree.idi] = [cur_path, tree.data]
    return
  create_roles(tree.left, roles, cur_path + ['L'])
  create_roles(tree.right, roles, cur_path + ['R'])

'''
def inorder(tree):
  if tree:
    inorder(tree.left)
    #if not tree.left and not tree.right:
    print(tree.data)
    inorder(tree.right)
'''
def gen_treerole(tokens, binaryParse):
  sent_tokens = binaryParse.split()
  sent_ids = [-1] * len(sent_tokens)
  idi = 0
  for ti, token in enumerate(sent_tokens):
    if token != '(' and token != ')':
      sent_ids[ti] = idi
      idi += 1
  btree = create_btree(sent_tokens, sent_ids)
  roles = {}
  create_roles(btree, roles, [])
  try:
    role_ids = match_bert_tokens(tokens, roles)
    roles_final = [ ''.join(roles[rid][0]) for rid in role_ids]
  except:
    # cases like 'can', 'not' in parse sentence
    roles_final = [ ''.join(roles[rid][0]) for rid in range(len(roles))]
    roles_final = roles_final[0:len(tokens)]
  if len(roles_final) != len(tokens):
    prev_len = len(roles_final)
    roles_final = roles_final + ['[SEP]'] * (len(tokens)-prev_len)
  return roles_final

def match_bert_tokens(bert_tokens, actual_tokens):
  role_ids, ati, bti = [], 0, 0
  atok = ''
  while ati < len(actual_tokens):
    atok = actual_tokens[ati][1]
    if atok == "``":
      atok = '"'
    if atok == "''":
      atok = '"'
    toki = 0
    while toki < len(atok):
      btok = bert_tokens[bti]
      if btok.startswith('##'):
        btok = btok[2:]
      assert(btok==atok[toki:toki+len(btok)].replace('`','"').lower())
      toki += len(btok)
      bti += 1
      role_ids.append(ati)
    ati += 1
  return role_ids

'''
s = "( ( A boy ) ( ( is ( ( jumping ( on skateboard ) ) ( in ( ( the middle ) ( of ( a ( red bridge ) ) ) ) ) ) ) . ) )"
print(gen_treerole(s))
sent_tokens = s.split()
sent_ids = [-1] * len(sent_tokens)
idi = 0
for ti, token in enumerate(sent_tokens):
  if token != '(' and token != ')':
    sent_ids[ti] = idi
    idi += 1
print(sent_tokens)
print(sent_ids)
btree = create_btree(sent_tokens, sent_ids)
roles = {}
create_roles(btree, roles, [])
print(roles)
#inorder(btree)
'''

# generate random tree
class RandNode:
  def __init__(self):
    self.left, self.right = None, None
def create_rand_roles(tree, roles, cur_path):
  if not tree:
    roles.append(cur_path)
    return
  create_rand_roles(tree.left, roles, cur_path + ['L'])
  create_rand_roles(tree.right, roles, cur_path + ['R'])
import random
def gen_rand_tree(num_tokens):
  assert(num_tokens != 0)
  if num_tokens == 1:
    return ['[UNK]']
  root = RandNode()
  free_edges = [(root, 0), (root, 1)]
  for i in range(num_tokens-2):
    rand_edge = random.choice(free_edges)
    node, child_id = rand_edge
    new_node = RandNode()
    if child_id == 0:
      node.left = new_node
    else:
      node.right = new_node
    free_edges.extend([(new_node, 0), (new_node, 1)])
    free_edges.remove(rand_edge)
  roles = []
  create_rand_roles(root, roles, [])
  return [ ''.join(role) for role in roles]










