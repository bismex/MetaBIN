#!/bin/bash 
#SBATCH -p part1 
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH -o ../../../sbatch/%x.txt 
#SBATCH -e ../../../sbatch/err_%x.txt 
#SBATCH --gres=gpu 
  
cd ../../../ 
source activate seokeon_torch16 
python3 ./tools/train_net.py --config-file ./configs/MetaBIN7/xx10.yml --resume 
source deactivate 
  
