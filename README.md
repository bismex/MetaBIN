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
```
- Need 10.1 cuda version or higher
- Need recent GPU drivers
- (optional) `cd fastreid/evaluation/rank_cylib; make all`

## 3) Check repository structure
```
DG-person
├── configs
├── datasets (*need to download and connect it by symbolic link [check section 4]*)
│   ├── *DukeMTMC-reID
│   ├── *Market-1501-v15.09.15
│   ├── *MSMT17_V2
│   ├── *vehicleid
│   ├── *veri
│   ├── *veri_keypoint
│   ├── *VERI-Wild
├── demo
├── fastreid
├── matlab
├── tests
├── tools
├── *pretrained (*need to make logs folder and connect it by symbolic link [check section 5]*)
├── *logs (*need to make logs folder and connect it by symbolic link [check section 5]*)
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

