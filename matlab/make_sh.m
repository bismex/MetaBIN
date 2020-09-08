function make_sh(find_file, conda_id, name_command, add_name_command, num_cpu, num_gpu, save_folder, add_txt_name)

% input parameters
% find_file = 'Augmentation';
% conda_id = 'seokeon_reid';
% name_command = 'python3 ./tools/train_net.py --config-file ./configs/Augmentation/';
% add_name_command = '--resume';
% num_gpu = 1; 
% save_folder = 'new';
% add_txt_name = '';

% fixed parameters
old_name_format = 'yml';
new_name_format = 'sh';
conda_on = 0; % 1:conda 0:source
% num_cpu = 2;
num_root = 2; % based on yml file

% processing
num_sub = num_root;
if ~isempty(save_folder)
    num_root = num_root + 1; % because sh file is saved in the save_folder
end
all_file_name = dir(fullfile(pwd, find_file, ['*.', old_name_format]));
cell_folder_name = strsplit(all_file_name(1).folder, '/');
if num_root == 0
    run_folder = './';
else
    run_folder = '';
    for j = 1 : num_root
        run_folder = [run_folder, '../']; 
    end
end
if num_gpu == 1
    name_gpu = 'gpu';
else
    name_gpu = ['gpu:', num2str(num_gpu)];
end
if conda_on
    conda_command = 'conda';
else
    conda_command = 'source';
end



% fprintf(['---------------------------------------------\n']);
% fprintf(['load format: ', old_name_format, '\n']);
% fprintf(['save folder: ', save_folder, '\n']);
% fprintf(['save format: ', new_name_format, '\n']);
% fprintf(['run folder: ', run_folder, '\n']);
% fprintf(['added text in name: ', add_txt_name, '\n']);
% fprintf(['---------------------------------------------\n']);
% fprintf(['num cpu: ', num2str(num_cpu), '\n']);
% fprintf(['num gpu: ', num2str(num_gpu), '\n']);
% fprintf(['name gpu: ', name_gpu, '\n']);
% fprintf(['conda id: ', conda_id, '\n']);
% fprintf(['conda command: ', conda_command, '\n']);
% fprintf(['main command: ', name_command, '$(',old_name_format,' name)$ \n']);
% fprintf(['added txt in command: ', add_name_command, '\n']);
fprintf(['=============================================\n']);


find_file_name = ['$(',old_name_format,'_name)$'];

cell_all_command = {};
cell_all_command = cat(1, cell_all_command, '#!/bin/bash');
cell_all_command = cat(1, cell_all_command, '#SBATCH -p part1');
cell_all_command = cat(1, cell_all_command, '#SBATCH -N 1');
cell_all_command = cat(1, cell_all_command, ['#SBATCH -n ', num2str(num_cpu)]);
cell_all_command = cat(1, cell_all_command, ['#SBATCH -o ', run_folder, 'sbatch/%%x.txt']);
cell_all_command = cat(1, cell_all_command, ['#SBATCH -e ', run_folder, 'sbatch/err_%%x.txt']);
cell_all_command = cat(1, cell_all_command, ['#SBATCH --gres=', name_gpu]);
cell_all_command = cat(1, cell_all_command, ' ');
cell_all_command = cat(1, cell_all_command, ['cd ', run_folder]);
cell_all_command = cat(1, cell_all_command, [conda_command, ' activate ', conda_id]);
cell_all_command = cat(1, cell_all_command, [name_command, find_file_name, ' ', add_name_command]);
cell_all_command = cat(1, cell_all_command, [conda_command, ' deactivate']);
cell_all_command = cat(1, cell_all_command, ' ');

idx_file = 0;
for j = 1 : length(cell_all_command)
    if ~isempty(strfind(cell_all_command{j}, find_file_name))
        idx_file = j;
    end
end
if idx_file == 0
    fprintf('error in idx_file\n')
end


if ~isdir(fullfile(pwd, save_folder))
    fprintf(['make folder: ', save_folder, '\n']);
    mkdir(fullfile(pwd, save_folder));
end

for i = 1 : length(all_file_name)
    old_file_name = all_file_name(i).name;
    new_file_name = [old_file_name(1:strfind(old_file_name, old_name_format)-2), add_txt_name, '.', new_name_format];
%     fprintf(
    
    fileID = fopen(fullfile(pwd, save_folder, new_file_name),'w');
    for j = 1 : length(cell_all_command)
        if j == idx_file
            idx_start = strfind(cell_all_command{j}, find_file_name);
            idx_end = strfind(cell_all_command{j}, find_file_name) + length(find_file_name)-1;
            new_command = [cell_all_command{j}(1:idx_start-1), old_file_name, cell_all_command{j}(idx_end + 1:end)];
            fprintf(fileID, [new_command, ' \n']);
            if i == 1
                fprintf([new_command, ' \n']);
            end
        else
            fprintf(fileID, [cell_all_command{j}, ' \n']);
            if i == 1
                fprintf([cell_all_command{j}, ' \n']);
            end
        end
    end
    if i == 1
        fprintf(['=============================================\n']);
    end
    fclose(fileID);
    fprintf(['(',num2str(i), '/',num2str(length(all_file_name)),') Make file "', old_file_name, '" -> "', fullfile(save_folder, new_file_name), '"\n'])
    
end
