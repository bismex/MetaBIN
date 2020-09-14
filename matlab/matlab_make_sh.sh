matlab -nodisplay -r "make_sh('Baseline', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Baseline/', '', 1, 1, 'Baseline/cpu1', ''); make_sh('Baseline', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Baseline/', '', 2, 1, 'Baseline/cpu2', ''); make_sh('Baseline', 'seokeon_torch16', 'python3 ./tools/train_net.py --config-file ./configs/Baseline/', '--resume', 2, 1, 'Baseline/resume', ''); exit" 
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
