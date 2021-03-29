**Feel free to visit my [homepage](https://sites.google.com/site/seokeonchoi/) and [awesome person re-id github page](https://github.com/bismex/Awesome-person-re-identification)**

---


## Meta Batch-Instance Normalization for Generalizable Person Re-Identification [CVPR2021 paper]

---


# MetaBIN

`git clone our_repository`
- If you can't clone our repository, you can download it in this [[link](https://drive.google.com/u/0/uc?id=1hcHh52TR6glihkOvfVKBewXbtPPKWSEZ&export=download)]

## 1) Prerequisites

- Ubuntu 18.04
- Python 3.6
- Pytorch 1.7+
- NVIDIA GPU (>=8,000MiB)
- Anaconda 4.8.3
- CUDA 10.1 (optional)
- Recent GPU driver (Need to support AMP [[link](https://pytorch.org/docs/stable/amp.html)])


## 2) Preparation

```
conda create -n MetaBIN python=3.6
conda activate MetaBIN
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
pip install tensorboard
pip install Cython
pip install yacs
pip install termcolor
pip install tabulate
pip install scikit-learn

pip install h5py
pip install imageio
pip install openpyxl 
pip install matplotlib 
pip install pandas 
pip install seaborn
```

## 3) Test only
  
- Download our model [[link](https://drive.google.com/u/0/uc?id=1hg4cYTFg5fZKqXIdUWOiO3P1xoSJtG3M&export=download)] to `MetaBIN/logs/Sample/DG-mobilenet`
```
├── MetaBIN/logs/Sample/DG-mobilenet
│   ├── last_checkpoint
│   ├── model_0099999.pth
│   ├── result.png
```

- Download test datasets [[link](https://drive.google.com/u/0/uc?id=1sKTv52366vE7zpdJYQsFW062_6BxDgOy&export=download)] to `MetaBIN/datasets/`
```
├── MetaBIN/datasets
│   ├── GRID
│   ├── prid_2011
│   ├── QMUL-iLIDS
│   ├── viper
```

- Execute run_file
`cd MetaBIN/`
`sh run_evaluate.sh`

- you can get the following results

| Datasets                  | Rank-1   | Rank-5   | Rank-10   | mAP    | mINP   | TPR@FPR=0.0001   | TPR@FPR=0.001   | TPR@FPR=0.01   |
|:--------------------------|:---------|:---------|:----------|:-------|:-------|:-----------------|:----------------|:---------------|
| ALL_GRID_average          | 49.68%   | 67.52%   | 76.80%    | 58.10% | 58.10% | 0.00%            | 0.00%           | 46.35%         |
| ALL_GRID_std              | 2.30%    | 3.56%    | 3.14%     | 2.58%  | 2.58%  | 0.00%            | 0.00%           | 26.49%         |
| ALL_VIPER_only_10_average | 56.90%   | 76.71%   | 82.03%    | 65.98% | 65.98% | 0.00%            | 0.00%           | 50.97%         |
| ALL_VIPER_only_10_std     | 2.97%    | 2.11%    | 2.06%     | 2.35%  | 2.35%  | 0.00%            | 0.00%           | 8.45%          |
| ALL_PRID_average          | 72.50%   | 88.20%   | 91.30%    | 79.78% | 79.78% | 0.00%            | 0.00%           | 91.00%         |
| ALL_PRID_std              | 2.20%    | 2.60%    | 2.00%     | 1.88%  | 1.88%  | 0.00%            | 0.00%           | 1.47%          |
| ALL_iLIDS_average         | 79.67%   | 93.33%   | 97.33%    | 85.51% | 85.51% | 0.00%            | 0.00%           | 56.13%         |
| ALL_iLIDS_std             | 4.40%    | 2.47%    | 2.26%     | 2.80%  | 2.80%  | 0.00%            | 0.00%           | 15.77%         |
| ** all_average **         | 64.69%   | 81.44%   | 86.86%    | 72.34% | 72.34% | 0.00%            | 0.00%           | 61.11%         |


- Other models [[link](https://drive.google.com/u/0/uc?id=1-PcLQyNJSiL4h7gYvHNk2bpJN4PkT92R&export=download)]

---

# Advanced (train new models)

## 4) Check the below repository structure
```
MetaBIN/
├── configs/
├── datasets/ (*need to download and connect it by symbolic link [check section 4], please check the folder name*)
│   ├── *cuhk02
│   ├── *cuhk03
│   ├── *CUHK-SYSU
│   ├── *DukeMTMC-reID
│   ├── *GRID
│   ├── *Market-1501-v15.09.15
│   ├── *prid_2011
│   ├── *QMUL-iLIDS
│   ├── *viper
├── demo/
├── fastreid/
├── logs/ 
├── pretrained/ 
├── tests/
├── tools/
'*' means symbolic links which you make (check below sections) 
```

## 5) download dataset and connect it

- Download dataset
  - For single-source DG
    - Need to download Market1501, DukeMTMC-REID [check section 8-1,2]
  - For multi-source DG
    - Training: Market1501, DukeMTMC-REID, CUHK02, CUHK03, CUHK-SYSU [check section 8-1,2,3,4,5]
    - Testing: GRID, PRID, QMUL i-LIDS, VIPer [check section 8-6,7,8,9]

- Symbolic link (recommended)
  - Check `symbolic_link_dataset.sh`
  - Modify each directory (need to change)
  - `cd MetaBIN`
  - `bash symbolic_link_dataset.sh`
  
- Direct connect (not recommended)
  - If you don't want to make symbolic link, move each dataset folder into `./datasets/`
  - Check the folder name for each dataset

## 6) Create pretrained and logs folder

- Symbolic link (recommended)
  - Make 'MetaBIN(logs)' and 'MetaBIN(pretrained)' folder outside MetaBIN
```
├── MetaBIN
│   ├── configs/
│   ├── ....
│   ├── tools/
├── MetaBIN(logs)
├── MetaBIN(pretrained)
```
  - `cd MetaBIN`
  - `bash symbolic_link_others.sh`
  - Download pretrained models and change name 
    - mobilenetv2_x1_0: [[link](https://mega.nz/#!NKp2wAIA!1NH1pbNzY_M2hVk_hdsxNM1NUOWvvGPHhaNr-fASF6c)]
    - mobilenetv2_x1_4: [[link](https://mega.nz/#!RGhgEIwS!xN2s2ZdyqI6vQ3EwgmRXLEW3khr9tpXg96G9SUJugGk)]
    - change name as `mobilenetv2_1.0.pth`, `mobilenetv2_1.4.pth`
  - Or download pretrained models [[link](https://drive.google.com/u/0/uc?id=1o-MqjM1YBeUoZB5mNlGiB6LVSIA2RV71&export=download)]

- Direct connect (not recommended)
  - Make 'pretrained' and 'logs' folder in `MetaBIN`
  - Move the pretrained models to `pretrained`
  

## 7) Train

- If you run code in pycharm
  - tools/train_net.py -> Edit congifuration
  - Working directory: `your folders/MetaBIN/`
  - Parameters: `--config-file ./configs/Sample/DG-mobilenet.yml`

- Single GPU

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml`

- Single GPU (specific GPU)

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml MODEL.DEVICE "cuda:0"`

- Multiple GPUs

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --num-gpus 2`

- Resume (model weights is automatically loaded based on `last_checkpoint` file in logs)

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --resume`

- Evaluation only

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --eval-only`


## 8) Datasets

- (1) Market1501
  - Create a directory named `Market-1501-v15.09.15`
  - Download the dataset to `Market-1501-v15.09.15` from [link](http://www.liangzheng.org/Project/project_reid.html) and extract the files.
  - The data structure should look like
  ```
  Market-1501-v15.09.15/
  ├── bounding_box_test/
  ├── bounding_box_train/
  ├── gt_bbox/
  ├── gt_query/
  ├── query/
  ```

- (2) DukeMTMC-reID
  - Create a directory called `DukeMTMC-reID`
  - Download `DukeMTMC-reID` from [link](http://vision.cs.duke.edu/DukeMTMC/) and extract the files.
  - The data structure should look like
  ```
  DukeMTMC-reID/
  ├── bounding_box_test/
  ├── bounding_box_train/
  ├── query/
  ```

- (3) CUHK02
  - Create `cuhk02` folder
  - Download the data from [link](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and put it under `cuhk02`.
    - The data structure should look like
  ```
  cuhk02/
  ├── P1/
  ├── P2/
  ├── P3/
  ├── P4/
  ├── P5/
  ```
  
- (4) CUHK03
  - Create `cuhk03` folder
  - Download dataset to `cuhk03` from [link](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and extract “cuhk03_release.zip”, resulting in “cuhk03/cuhk03_release/”.
  - Download the new split (767/700) from person-re-ranking. What you need are “cuhk03_new_protocol_config_detected.mat” and “cuhk03_new_protocol_config_labeled.mat”. Put these two mat files under `cuhk03`.
  - The data structure should look like
  ```
  cuhk03/
  ├── cuhk03_release/
  ├── cuhk03_new_protocol_config_detected.mat
  ├── cuhk03_new_protocol_config_labeled.mat
  ```
  
- (5) Person Search (CUHK-SYSU)
  - Create a directory called `CUHK-SYSU`
  - Download `CUHK-SYSU` from [link](https://github.com/ShuangLI59/person_search) and extract the files.
  - Cropped images can be created by my matlab code `make_cropped_image.m`
  - The data structure should look like
  ```
  CUHK-SYSU/
  ├── annotation/
  ├── Image/
  ├── cropped_image/
  ├── make_cropped_image.m (my matlab code)
  ```


- (6) GRID
  - Create a directory called `GRID`
  - Download `GRID` from [link](http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip) and extract the files.
  - Split sets (`splits.json`) can be created by python code `grid.py`
  - The data structure should look like

  ```
  GRID/
  ├── gallery/
  ├── probe/
  ├── splits_single_shot.json (This will be created by `grid.py` in `fastreid/data/datasets/` folder)
  ```

  
- (7) PRID
  - Create a directory called `prid_2011`
  - Download `prid_2011` from [link](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/) and extract the files.
  - Split sets (`splits_single_shot.json`) can be created by python code `prid.py`
  - The data structure should look like

  ```
  prid_2011/
  ├── single_shot/
  ├── multi_shot/
  ├── splits_single_shot.json (This will be created by `prid.py` in `fastreid/data/datasets/` folder)
  ```
  
  
- (8) QMUL i-LIDS
  - http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz
  - https://github.com/BJTUJia/person_reID_DualNorm
  - Create a directory called `QMUL_iLIDS`
  - Download `QMUL_iLIDS` from the upper links
  - Split sets () can be created by python code `iLIDS.py`
  - The data structure should look like

  ```
  QMUL-iLIDS/
  ├── images/
  ├── splits.json (This will be created by `iLIDS.py` in `fastreid/data/datasets/` folder)
  ```

- (9) VIPer
  - Create a directory called `viper`
  - Download `viper` from [link](https://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip) and extract the files.
  - Split sets can be created by my matlab code `make_split.m`
  - The data structure should look like
  ```
  viper/
  ├── cam_a/
  ├── cam_b/
  ├── make_split.m (my matlab code)
  ├── split_1a # Train: split1, Test: split2 ([query]cam1->[gallery]cam2)
  ├── split_1b # Train: split2, Test: split1 (cam1->cam2)
  ├── split_1c # Train: split1, Test: split2 (cam2->cam1)
  ├── split_1d # Train: split2, Test: split1 (cam2->cam1)
  ...
  ...
  ├── split_10a
  ├── split_10b
  ├── split_10c
  ├── split_10d
  ```

## 9) Code structure


- Our code is based on fastreid [link](https://github.com/JDAI-CV/fast-reid)
  
- fastreid/config/defaults.py: default settings (parameters)
- fastreid/data/datasets/: about datasets

- tools/train_net.py: Main code (train/test/tsne/visualize)
- fastreid/engine/defaults.py: build dataset, build model
  - fastreid/data/build.py: build datasets (base model/meta-train/meta-test)
  - fastreid/data/samplers/triplet_sampler.py: data sampler
  - fastreid/modeling/meta_arch/metalearning.py: build model
    - fastreid/modeling/backbones/mobilenet_v2.py or resnet.py: backbone network
    - fastreid/heads/metalearning_head.py: head network (bnneck)
  - fastreid/solver/build.py: build optimizer and scheduler
- fastreid/engine/train_loop.py: main train code
  - run_step_meta_learning1(): update base model
  - run_step_meta_learning2(): update balancing parameters (meta-learning)


## 10) Handling errors

- AMP
  - If the version of your GPU driver is old, you cannot use AMP(automatic mixed precision).
  - If so, modify the AMP option to False in `/MetaBIN/configs/Sample/DG-mobilenet.yml` 
  - The memory usage will increase.
- Fastreid evaluation
  - If a compile error occurs in fastreid, run the following command.
  - `cd fastreid/evaluation/rank_cylib; make all`
- No such file or directory 'logs/Sample'
  - Please check `logs` (section 3)
- No such file or directory 'pretrained'
  - Please check `pretrained` (section 6)
- No such file or directory 'datasets'
  - Please check `datasets` (section 8)




## Citation
```
@InProceedings{choi2021metabin,
title = {Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared Person Re-Identification},
author = {Choi, Seokeon and Kim, Taekyung and Jeong, Minki and Park, Hyoungseob and Kim, Changick},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

