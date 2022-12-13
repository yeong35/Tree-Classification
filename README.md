# Tree-Classification
We customize [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) for the binary tree classification.

We add tree dataloader for the classification and [pointnet_demo.ipynb](./pointnet_demo.ipynb) for colab users.

## Table of contents
- [Environment Setting](#environment-setting)

- [Usage](#usage)

- [Demo](#demo)

## Environment Setting
If you want to train this model, you should install below library. If you don't want to install libraries, you can use our [demo file](./pointnet_demo.ipynb) too.
- python 3.9.12
- torch 1.0.2
- sklearn 1.10.1+cu102
```
pip install torch
pip install sklearn
```

## Usage
You can use .off or .ply but My code will convert your .off files to .ply files. If you don't want to change it, give convert_off_to_ply parameter to False
## Dataset
If you want to make your own dataset, change paths of dataset (larch, pine) and run [train_valid_maker.py](./train_valid_maker.py)

Also, add id.txt to misc directory for your tast. Here is [example](tree_id.txt)
```
python train_valid_maker.py
```
## Train model
```
cd ./utils
python train_classification.py --batchSize [batch size] --num_points [number of points for each data] --nepoch [training epoch] --dataset [dataset path] --dataset_type [dataset type]
```

example
```
cd ./utils
python train_classification.py --dataset ./dataset --dataset_type tree
```


## Test model
```
cd ./utils
python train_classification.py --batchSize [batch size] --num_points [number of points for each data] --nepoch [training epoch] --dataset [dataset path] --dataset_type [dataset type] -train False
```

example
```
cd ./utils
python train_classification.py --dataset ./dataset --dataset_type tree -train False
```

## Demo
If you want to run demo file, you can use [pointnet_demo.ipynb](./pointnet_demo.ipynb).

## Project Period
Oct 24, 2022 - Nov 30, 2022