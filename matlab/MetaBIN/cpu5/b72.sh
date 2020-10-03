#!/bin/bash 
#SBATCH -p part1 
#SBATCH -N 1 
#SBATCH -n 5 
#SBATCH -o ../../../sbatch/%x.txt 
#SBATCH -e ../../../sbatch/err_%x.txt 
#SBATCH --gres=gpu 
  
cd ../../../ 
source activate seokeon_torch16 
python3 ./tools/train_net.py --config-file ./configs/MetaBIN/b72.yml  
source deactivate 
  
