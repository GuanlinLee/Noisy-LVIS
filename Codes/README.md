

<h1> Tested with </h1>
<div>
 <ul>
  <li>python==3.8.12</li>
  <li>torch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>mmdet==2.21.0</li>
  <li>lvis</li>
  <li>Tested on CUDA 10.2 and RHEL 8 system</li>
</ul> 
</div>


<h1> Getting Started </h1>
Create a virtual environment

```
conda create --name mmdet pytorch=1.7.1 -y
conda activate mmdet
```

1. Install dependency packages
```
conda install torchvision -y
conda install pandas scipy -y
conda install opencv -y
```

1. Install MMDetection
```
pip install openmim
mim install mmdet==2.21.0
```
3. Clone this repo
```
git clone https://github.com/GuanlinLee/Noisy-LVIS.git
cd Noisy-LVIS
```
4. Create data directory, download COCO 2017 datasets at https://cocodataset.org/#download (2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB]) and extract the zip files:
```
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

#download and unzip LVIS annotations
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip

```
5. put the noisy annotations in the ./datasets/coco/annotations/

6. modify the data path in config files

<h1>Training</h1>
To Train on multiple GPUs use <i>tools/dist_train.sh</i> to launch training on multiple GPUs:

```
./tools/dist_train.sh ./configs/<experiment>/<variant.py> <#GPUs>
```


     
<h1> Acknowledgements </h1>
     This code uses the <a href='https://github.com/open-mmlab/mmdetection'>mmdet</a> framework. It also uses <a href='https://github.com/tztztztztz/eqlv2'>EQLv2</a>, <a href='https://github.com/timy90022/DropLoss'>DropLoss</a>, <a href='https://github.com/open-mmlab/mmdetection'>Seesaw Loss</a> and <a href='https://github.com/kostas1515/GOL'>GOL</a>. Thank you for your wonderfull works! 
     

     
