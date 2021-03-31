clc
clear
gap = 0; % 0: MATLAB, 1: c++/python
print_num = 50;

save_dir = 'cropped_image';
mkdir(save_dir)

person_all = load('./annotation/Person.mat');
person_all = person_all.Person;

cnt_total = 0;
for i=1:length(person_all)
    cnt_total = cnt_total + person_all(i).nAppear;
end

cnt = 0;
for i=1:length(person_all)
    for j=1:person_all(i).nAppear
    
        savename = [person_all(i).idname(1),  num2str(str2num(person_all(i).idname(2:end)), '%05.f'), '_n', num2str(j, '%02.f'), '_s', num2str(str2num(person_all(i).scene(j).imname(2:end-4)), '%05.f') , '_hard', num2str(person_all(i).scene(j).ishard), '.png'];
        im_dir = fullfile('Image', 'SSM' , person_all(i).scene(j).imname);
        im = imread(im_dir);

        xmin = person_all(i).scene(j).idlocate(1) + gap;
        ymin = person_all(i).scene(j).idlocate(2) + gap;
        width = person_all(i).scene(j).idlocate(3);
        height = person_all(i).scene(j).idlocate(4);
        im_crop = im(ymin:ymin+height,xmin:xmin+width,:);

        imwrite(im_crop, fullfile(save_dir, savename));
        cnt = cnt + 1;
        
        
        if rem(cnt, print_num) == 0
            fprintf(['Save images: (', num2str(cnt), '/', num2str(cnt_total), ')\n'])
        end
    end
end