import json
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
import matplotlib.pyplot as plt

font = {
	"font.family": "Roboto",
	"font.size": 12,
}
paper_rc = {
	"lines.linewidth": 3,
	"lines.markersize": 10,
}

def plot(y, save):
	fig, ax = plt.subplots(
		ncols=1, nrows=1, constrained_layout=True, figsize=(12, 9)
	)
	ax.bar(range(len(y)), y)
	plt.xticks(fontsize=25)
	plt.yticks(fontsize=25)
	ax.set_xlim(0, 1203)
	plt.savefig(save)
	plt.close()

def unjson(file):
	with open(file, 'r') as fo:
		dict = json.load(fo)
	return dict


val_ann = unjson('lvis_v1_val.json')
cate = val_ann['categories']

SUPER_CLASS_SET = ['food', 'material', 'causal_agent', 'natural_object',
				   'living_thing', 'fixture', 'artical', 'way', 'block',
				   'padding', 'line', 'opening', 'decoration', 'commodity',
				   'strip', 'plaything', 'fabric', 'structure', 'sheet',
				   'surface', 'float', 'creation', 'instrumentality', 'organism',
				   'communication', 'arrangement', 'measure', 'others']
print('num of SUPER CLASS:', len(SUPER_CLASS_SET))
class_to_super_class = {'others':[]}
D = 6
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

id_to_super_class = {}

for k,v in class_to_super_class.items():
	for id in v:
		id_to_super_class[id] = k

print(id_to_super_class)
print(len(id_to_super_class))
train_ann = unjson('lvis_v1_train.json')



cate_counter = np.zeros(len(cate), dtype=np.int)
for i in range(len(train_ann['annotations'])):
	cate_counter[train_ann['annotations'][i]['category_id']-1] += 1

new_index = np.argsort(-cate_counter)
print(cate_counter[new_index].tolist())
plot(cate_counter[new_index].tolist(), 'train_distribution.png')

# r is noise rate
r = 0.4

count = 0
print(len((train_ann['annotations'])))
for i in range(len(train_ann['annotations'])):
	if np.random.random() < r:
		super_class = id_to_super_class[train_ann['annotations'][i]['category_id']]
		if super_class != 'others':
			new_cate = np.random.choice(class_to_super_class[super_class], 1, replace=False)[0]
			#print('gt:', train_ann['annotations'][i]['category_id'])
			#print('noise:', new_cate)
			train_ann['annotations'][i]['category_id'] = int(new_cate)
			count += 1


with open('lvis_v1_train_AN_%f.json'%r, 'w') as file:
	json.dump(train_ann, file)
print(len((train_ann['annotations'])))

print('total changed:', count)
print('change ratio:', count/len(train_ann['annotations']))

cate_counter = np.zeros(len(cate), dtype=np.int)
for i in range(len(train_ann['annotations'])):
	cate_counter[train_ann['annotations'][i]['category_id']-1] += 1

print(cate_counter[new_index].tolist())
plot(cate_counter[new_index].tolist(), 'train_distribution_AN_%f.png'%r)