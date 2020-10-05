# DG-person

## 1) git clone my_repository

`git clone ~~~~`

## 2) Preparation

```
conda create -n DG-person python=3.6
conda activate DG-person
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
pip install tensorboard
pip install Cython
pip install yacs
pip install termcolor
pip install tabulate
pip install scikit-learn
pip install h5py
pip install imageio
```
- Need 10.1 cuda version or higher
- Need recent GPU drivers
- (optional) `cd fastreid/evaluation/rank_cylib; make all`

## 3) Check repository structure
```
DG-person/
├── configs/
├── datasets/ (*need to download and connect it by symbolic link [check section 4]*)
│   ├── *DukeMTMC-reID
│   ├── *Market-1501-v15.09.15
│   ├── *MSMT17_V2
│   ├── *vehicleid
│   ├── *veri
│   ├── *veri_keypoint
│   ├── *VERI-Wild
├── demo/
├── fastreid/
├── matlab/
├── tests/
├── tools/
├── *pretrained/ (*need to make logs folder and connect it by symbolic link [check section 5]*)
├── *logs/ (*need to make logs folder and connect it by symbolic link [check section 5]*)
'*' means symbolic links which you make (check below sections) 
```

## 4) download dataset and connect it

- Download dataset [link]
- Symbolic link (recommended)
  - Check `symbolic_link_dataset.sh`
  - Modify each directory
  - `cd DG-person`
  - `bash symbolic_link_dataset.sh`
  
- Direct connect (not recommended)
  - If you don't want to make symbolic link, move each dataset folder into `./datasets/`
  - Check the name of dataset folders

## 5) Create pretrained and logs folder

- Symbolic link (recommended)
  - Make 'pretrained' and 'logs' folder 
  ```
  ├── DG-person
  ├── DG-person(logs)
  ├── DG-person(pretrained)
  ```
  - `cd DG-person`
  - `bash symbolic_link_others.sh`
  - Download pretrained models [link] and move them on the folder `DG-person/pretrained/`
  - Download pretrained models [link] and change name
    - mobilenetv2_x1_0 (top-1 71.3%): [[link](https://mega.nz/#!NKp2wAIA!1NH1pbNzY_M2hVk_hdsxNM1NUOWvvGPHhaNr-fASF6c)]
    - mobilenetv2_x1_4 (top-1 73.9%): [[link](https://mega.nz/#!RGhgEIwS!xN2s2ZdyqI6vQ3EwgmRXLEW3khr9tpXg96G9SUJugGk)]
    - change name as `mobilenetv2_1.0.pth`, `mobilenetv2_1.4.pth`

- Direct connect (not recommended)
  - Make 'pretrained' and 'logs' folder in `DG-person`
  - Move the pretrained models on `DG-person(pretrained)`
  
  
## 6) Train

- If you run code in pycharm
  - tools/train_net.py -> Edit congifuration
  - Working directory: `your folders/DG-person/`
  - Parameters: `--config-file ./configs/Sample/v00_person.yml`

- Single GPU

`python3 ./tools/train_net.py --config-file ./configs/Sample/v00_person.yml`


- Single GPU (specific GPU)

`python3 ./tools/train_net.py --config-file ./configs/Sample/v00_person.yml MODEL.DEVICE "cuda:0"`


- Multiple GPUs

`python3 ./tools/train_net.py --config-file ./configs/Sample/v00_person.yml --num-gpus 2`

- Resume (model weights is automatically loaded based on `last_checkpoint` file in logs)

`python3 ./tools/train_net.py --config-file ./configs/Sample/v00_person.yml --resume`

- Evaluation only

`python3 ./tools/train_net.py --config-file ./configs/Sample/v00_person.yml --eval-only`

## 7) Datasets

- CUHK03
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
  
- CUHK02
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
  
- Market1501
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
  

- DukeMTMC-reID
  - Create a directory called `DukeMTMC-reID`
  - Download `DukeMTMC-reID` from [link](http://vision.cs.duke.edu/DukeMTMC/) and extract the files.
  - The data structure should look like
  ```
  DukeMTMC-reID/
  ├── bounding_box_test/
  ├── bounding_box_train/
  ├── query/
  ```

- Person Search (CUHK-SYSU)
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

- VIPer
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


- GRID
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
  
  
- PRID
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
  
  
- i-LIDS
  - http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz
  - https://github.com/BJTUJia/person_reID_DualNorm
  - Create a directory called `QMUL_iLIDS`
  - Download `QMUL_iLIDS` from the upper links
  - Split sets () can be created by python code `iLIDS.py`
  - The data structure should look like
  
  ```
  QMUL-iLIDS/
  ├── images/
  
  
  ```