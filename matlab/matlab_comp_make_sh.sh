matlab -nodisplay -r "make_sh('Comparison', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Comparison/', '', 1, 1, 'Comparison/cpu1', ''); make_sh('Comparison', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Comparison/', '', 2, 1, 'Comparison/cpu2', ''); make_sh('Comparison', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Comparison/', '', 5, 1, 'Comparison/cpu5', ''); make_sh('Comparison', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Comparison/', '--resume', 5, 1, 'Comparison/resume5', ''); make_sh('Comparison', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Comparison/', '--resume', 1, 1, 'Comparison/resume1', ''); make_sh('Comparison', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Comparison/', '--resume', 2, 1, 'Comparison/resume2', ''); exit" 

#  make_sh('MetaReg', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/MetaReg/', '--resume', 1, 1, 'MetaReg/resume1', ''); make_sh('MetaReg', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/MetaReg/', '--resume', 3, 1, 'MetaReg/resume3', ''); make_sh('MetaReg', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/MetaReg/', '--resume', 5, 1, 'MetaReg/resume5', '');

#matlab -nodisplay -r "make_sh('METAv4', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/METAv3/', '', 2, 1, 'cpu2', ''); exit" 
#matlab -nodisplay -r "make_sh('METAv1', 'seokeon_reid', 'python3 ./tools/train_net.py --config-file ./configs/METAv1/', '--resume', 1, 'resume', 'r'); exit" 

# input parameters
# find_file = 'Augmentation'
# conda_id = 'seokeon_reid';
# name_command = python3 ./tools/train_net.py --config-file ./configs/Augmentation/;
# add_name_command = '';
# num_gpu = 1; 
# save_folder = 'gpu1';
# add_txt_name = '';