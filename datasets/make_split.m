clc
clear

cam_a = dir('cam_a/*.bmp');
cam_b = dir('cam_b/*.bmp');

% check id is ordered

for i=1:length(cam_a)
    name_split = split(cam_a(i).name, '_');
    cam_a_id = str2num(name_split{1});
    
    name_split = split(cam_b(i).name, '_');
    cam_b_id = str2num(name_split{1});
    
    if cam_a_id ~= cam_b_id
        fprintf(['[', num2str(i), '] cam_a_id and cam_b_id are different\n'])
    end
end


cam_a_new = {};
for i=1:length(cam_a)
    local = {};
    local.im_path = fullfile(cam_a(i).folder, cam_a(i).name);
    name_split = split(cam_a(i).name, '_');
    name_split2 = split(name_split{2}, '.');
    local.id = str2num(name_split{1});
    local.dir = str2num(name_split2{1});
    local.name = ['p', num2str(local.id, '%03.f'), '_c1_d', num2str(local.dir, '%03.f'), '.png'];
    cam_a_new = cat(1, cam_a_new, local);
end


cam_b_new = {};
for i=1:length(cam_b)
    local = {};
    local.im_path = fullfile(cam_b(i).folder, cam_b(i).name);
    name_split = split(cam_b(i).name, '_');
    name_split2 = split(name_split{2}, '.');
    local.id = str2num(name_split{1});
    local.dir = str2num(name_split2{1});
    local.name = ['p', num2str(local.id, '%03.f'), '_c2_d', num2str(local.dir, '%03.f'), '.png'];
    cam_b_new = cat(1, cam_b_new, local);
end


num_class = 632;

seed = 0;
rng(seed);
num = 10;


% for i = 1:num
%     folder_name = ['split_', num2str(i)];
%     num_all = randperm(num_class);
%     split1 = num_all(1:num_class/2);
%     split2 = num_all(num_class/2+1:end);
%     
%     
%     folder_name2 = [folder_name, 'a']; % Train: split1, Test: split2 ([query]cam1->[gallery]cam2)
%     mkdir(folder_name2);
%     
%     folder_name3 = fullfile(folder_name2, 'train');
%     mkdir(folder_name3);
%     write_image(split1, cam_a_new, folder_name3);
%     write_image(split1, cam_b_new, folder_name3);  
%     
%     folder_name3 = fullfile(folder_name2, 'query');
%     mkdir(folder_name3);
%     write_image(split2, cam_a_new, folder_name3);
%     
%     folder_name3 = fullfile(folder_name2, 'gallery');
%     mkdir(folder_name3);
%     write_image(split2, cam_b_new, folder_name3);
%     
%     
%     
%     folder_name2 = [folder_name, 'b']; % Train: split2, Test: split1 (cam1->cam2)
%     mkdir(folder_name2);
%     
%     folder_name3 = fullfile(folder_name2, 'train');
%     mkdir(folder_name3);
%     write_image(split2, cam_a_new, folder_name3);
%     write_image(split2, cam_b_new, folder_name3);  
%     
%     folder_name3 = fullfile(folder_name2, 'query');
%     mkdir(folder_name3);
%     write_image(split1, cam_a_new, folder_name3);
%     
%     folder_name3 = fullfile(folder_name2, 'gallery');
%     mkdir(folder_name3);
%     write_image(split1, cam_b_new, folder_name3);
%     
%     
%     
%     
%     folder_name2 = [folder_name, 'c']; % Train: split1, Test: split2 (cam2->cam1)
%     mkdir(folder_name2);
%     
%     folder_name3 = fullfile(folder_name2, 'train');
%     mkdir(folder_name3);
%     write_image(split1, cam_a_new, folder_name3);
%     write_image(split1, cam_b_new, folder_name3);  
%     
%     folder_name3 = fullfile(folder_name2, 'query');
%     mkdir(folder_name3);
%     write_image(split2, cam_b_new, folder_name3);
%     
%     folder_name3 = fullfile(folder_name2, 'gallery');
%     mkdir(folder_name3);
%     write_image(split2, cam_a_new, folder_name3);
%     
%     
%     
%     
%     folder_name2 = [folder_name, 'd']; % Train: split2, Test: split1 (cam2->cam1)
%     mkdir(folder_name2);
%     
%     folder_name3 = fullfile(folder_name2, 'train');
%     mkdir(folder_name3);
%     write_image(split2, cam_a_new, folder_name3);
%     write_image(split2, cam_b_new, folder_name3); 
%     
%     folder_name3 = fullfile(folder_name2, 'query');
%     mkdir(folder_name3);
%     write_image(split1, cam_b_new, folder_name3);
%     
%     folder_name3 = fullfile(folder_name2, 'gallery');
%     mkdir(folder_name3);
%     write_image(split1, cam_a_new, folder_name3);
%     
%     
%     fprintf(['total complete ', num2str(i), '\n'])
%     
% end


function write_image(split_num, image_cell, save_folder)

for j=1:length(split_num)
    imwrite(imread(image_cell{split_num(j)}.im_path), fullfile(save_folder, image_cell{split_num(j)}.name));
end
fprintf('complete\n')

end


