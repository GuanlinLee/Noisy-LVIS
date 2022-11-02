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
train_ann = unjson('lvis_v1_train.json')


cate_counter = np.zeros(len(cate), dtype=np.int)
for i in range(len(train_ann['annotations'])):
	cate_counter[train_ann['annotations'][i]['category_id']-1] += 1

new_index = np.argsort(-cate_counter)
print(cate_counter[new_index].tolist())
plot(cate_counter[new_index].tolist(), 'train_distribution.png')

# r is noise rate
r = 0.8

count = 0

for i in range(len(train_ann['annotations'])):
	if np.random.random() < r:
		new_cate = np.random.randint(1, len(cate))
		print('gt:', train_ann['annotations'][i]['category_id'])
		print('noise:', new_cate)
		train_ann['annotations'][i]['category_id'] = int(new_cate)
		count += 1

with open('lvis_v1_train_SN_%f.json'%r, 'w') as file:
	json.dump(train_ann, file)

print('total changed:', count)
print('change ratio:', count/len(train_ann['annotations']))

cate_counter = np.zeros(len(cate), dtype=np.int)
for i in range(len(train_ann['annotations'])):
	cate_counter[train_ann['annotations'][i]['category_id']-1] += 1

print(cate_counter[new_index].tolist())
plot(cate_counter[new_index].tolist(), 'train_distribution_SN_%f.png'%r)