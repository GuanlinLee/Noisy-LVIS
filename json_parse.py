import json
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
import pydot
def unjson(file):
	with open(file, 'r') as fo:
		dict = json.load(fo)
	return dict

tree_structure = {}
D = 6
counter_dict = {}
max_depth = 0
def class_counter(path_list):
	for path in path_list:
		p = path[0]
		depth = len(p) if len(p)<=D else D
		name = p[depth-1].name().split('.')[0]
		if name not in counter_dict.keys():
			counter_dict[name] = 1
		else:
			counter_dict[name] += 1
def build_tree_dict(path_list):
	global max_depth
	for path in path_list:
		p = path[0]
		depth = len(p) if len(p)<=D else D
		if len(p) > max_depth:
			max_depth = len(p)
		for i in range(depth):
			name = p[i].name().split('.')[0]
			if i == 0:
				sub_dict = create_dict(name, tree_structure)
			else:
				sub_dict = create_dict(name, sub_dict)
		if p[depth-1].name().split('.')[0] + '_' +str(counter_dict[p[depth-1].name().split('.')[0]]) not in sub_dict.keys():
				sub_dict[p[depth-1].name().split('.')[0] + '_' +str(counter_dict[p[depth-1].name().split('.')[0]])] = {'END':'END'}
def create_dict(name, d):
	if name not in d.keys():
		child_dict = {}
		d[name] = child_dict
	else:
		pass
	return d[name]
val_ann = unjson('lvis_v1_val.json')
cate = val_ann['categories']


synonyms_dict = {}
cate_id_map = {}
cate_counter = np.zeros(len(cate), dtype=np.int)
synonyms_set = []
for i in range(len(cate)):
	tmp = {}
	tmp['synonyms'] = cate[i]['synonyms']
	tmp['id'] = cate[i]['id']
	synonyms_dict[cate[i]['name']] = tmp
	cate_id_map[i+1] = cate[i]['name']
	try:
		hyper = wordnet.synset(cate[i]['synset']).hypernym_paths()
		synonyms_set.append(hyper)
	except:
		pass


class_counter(synonyms_set)
build_tree_dict(synonyms_set)


def draw(parent_name, child_name):
	edge = pydot.Edge(parent_name, child_name)
	graph.add_edge(edge)

def visit(node, parent=None):
	for k,v in node.items():
		if isinstance(v, dict):
			# We start with the root node whose parent is None
			# we don't want to graph the None node
			if parent:
				draw(parent, k)
			visit(v, k)
		else:
			pass
			#draw(parent, k)
			# drawing the label using a distinct name
			#draw(k, v)

graph = pydot.Dot(graph_type='graph', simplify=True)
visit(tree_structure)
#print(graph.get_nodes())
#exit(0)
#graph.del_node('END')
graph.write_png('example1_graph.png')

C = 0
for k,v in counter_dict.items():
	C += v


train_ann = unjson('lvis_v1_train.json')['annotations']
for i in range(len(train_ann)):
	cate_counter[train_ann[i]['category_id']-1] += 1



SUPER_CLASS_SET = ['food', 'material', 'causal_agent', 'natural_object',
				   'living_thing', 'fixture', 'artical', 'way', 'block',
				   'padding', 'line', 'opening', 'decoration', 'commodity',
				   'strip', 'plaything', 'fabric', 'structure', 'sheet',
				   'surface', 'float', 'creation', 'instrumentality', 'organism',
				   'communication', 'arrangement', 'measure', 'others' ]
print('num of SUPER CLASS:', len(SUPER_CLASS_SET))
class_to_super_class = {'others':[]}
for i in range(len(cate)):
	FLAG = False
	try:
		hyper = wordnet.synset(cate[i]['synset']).hypernym_paths()
		p = hyper[0]
		depth = len(p) if len(p) <= D else D
		for j in range(depth):
			name = p[j].name().split('.')[0]
			if (name in SUPER_CLASS_SET) and (not FLAG):
				if name not in class_to_super_class.keys():
					class_to_super_class[name] = [cate[i]['id']]
				else:
					class_to_super_class[name].append(cate[i]['id'])
				FLAG = True
		if not FLAG:
			class_to_super_class['others'].append(cate[i]['id'])
	except:
		class_to_super_class['others'].append(cate[i]['id'])



CC = 0
for k,v in class_to_super_class.items():
	print(k, len(v))
	CC += len(v)

print('total covert:', CC)

print('max depth:', max_depth)

id_to_fre = {}
for i in range(len(cate)):
	id_to_fre[cate[i]['id']] = cate[i]['frequency']

superclass_fre = {}
for k,v in class_to_super_class.items():
	superclass_fre[k] = {'r':0, 'c':0, 'f':0}
	for id in v:
		superclass_fre[k][id_to_fre[id]] += 1

sum = {'r':0, 'c':0, 'f':0}

for k, v in class_to_super_class.items():
	print(k,'\n')
	for f in ['r', 'c', 'f']:
		print(f, superclass_fre[k][f])
		sum[f] += superclass_fre[k][f]
	print('\n')

print(sum)