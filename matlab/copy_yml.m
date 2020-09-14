function copy_yml(copy_target, idx1, idx2, varargin)

fprintf('You should input like this(required): copy_yml("target_file_name", idx1, idx2)\n')
fprintf('You should input like this(recommended): copy_yml("target_file_name", idx1, idx2, num_pad, num_location)\n')

% copy_target = 'test1_veri.yml';
var_make = [idx1:idx2]; % which number you want to make
num_location = 1; % which index? ex> name15_25_30.txt -> if 1 -> find '15'
num_pad = 2; % ex> when num_pad = 2, output 01, 02, 03, ... 99
if nargin > 3
    num_pad = varargin{1};
end
fprintf(['num_par is set to ', num2str(num_pad), '\n'])
if nargin > 4
   num_location = varargin{2};
end
fprintf(['num_location is set to ', num2str(num_location), '\n'])

if isfile(copy_target)
    for i = 1:length(var_make)
        in_num = regexp(copy_target, '\d*', 'Match');
        str_num = in_num{num_location};

        start_idx = strfind(copy_target, str_num);
        end_idx = strfind(copy_target, str_num) + length(str_num)-1;


        if start_idx == 1
            out_name1 = '';
        else
            out_name1 = copy_target(1:start_idx-1);
        end

        if end_idx == length(copy_target)
            out_name2 = '';
        else
            out_name2 =  copy_target(end_idx+1:end);
        end

        out_num = var_make(i);
        final_name = [out_name1, num2str(out_num,['%0', num2str(num_pad), '.f']), out_name2];
        copyfile(copy_target, final_name);
        fprintf(['(',num2str(i), '/',num2str(length(var_make)),') Copy file "', copy_target, '" -> "', final_name, '"\n'])
    end
else
    fprintf([copy_target, ' does not exist\n'])
end
