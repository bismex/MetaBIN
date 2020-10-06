import os
import json
from json import JSONDecoder, JSONDecodeError
import re
import argparse


def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        NOT_WHITESPACE = re.compile(r'[^\s]')
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_dir', default='../logs')
    parser.add_argument('--folder_name', default='Sample2')
    parser.add_argument('--start_name', default='b20')
    parser.add_argument('--end_name', default='b31')
    parser.add_argument('--max_only', default=1)

    args = parser.parse_args()
    folder_dir = args.folder_dir
    folder_name = args.folder_name
    start_name = args.start_name
    end_name = args.end_name

    txt_name = start_name[0]
    start_num = int(start_name[1:])
    end_num = int(end_name[1:])

    main_txt = "** all_average **/Rank-1"

    find_txt = []
    find_txt.append("** all_average **/Rank-1")
    find_txt.append("ALL_GRID_average/Rank-1")
    find_txt.append("ALL_PRID_average/Rank-1")
    find_txt.append("ALL_VIPER_only_10_average/Rank-1")
    find_txt.append("ALL_iLIDS_average/Rank-1")
    find_txt.append("iteration")

    for i in range(start_num, end_num + 1):
        case = txt_name + str(i).zfill(2)
        file_name = os.path.join(folder_dir, folder_name, case, 'metrics.json')
        # print(case)
        if os.path.isfile(file_name):
            file = open(file_name, "r", encoding='utf-8')
            all_dict = dict()
            for obj in decode_stacked(file.read()):
                if main_txt in obj.keys():
                    for find_name in find_txt:
                        if find_name in obj.keys():
                            if find_name in all_dict.keys():
                                all_dict[find_name].append(obj[find_name])
                            else:
                                all_dict[find_name] = []
                                all_dict[find_name].append(obj[find_name])
            if len(all_dict) > 0:

                if args.max_only == 1:
                    max_val = max(all_dict[main_txt])
                    max_idx = [i for i, x in enumerate(all_dict[main_txt]) if x == max_val][0]
                    try:
                        acc_G = all_dict["ALL_GRID_average/Rank-1"][max_idx]
                    except:
                        acc_G = 0
                    try:
                        acc_P = all_dict["ALL_PRID_average/Rank-1"][max_idx]
                    except:
                        acc_P = 0
                    try:
                        acc_V = all_dict["ALL_VIPER_only_10_average/Rank-1"][max_idx]
                    except:
                        acc_V = 0
                    try:
                        acc_I = all_dict["ALL_iLIDS_average/Rank-1"][max_idx]
                    except:
                        acc_I = 0
                    print('[{}] (iter:{}, avg:{}, G:{}, P:{}, V:{}, I:{} ), final_iter:{}, eta:{}hours'.
                          format(case,
                                 round(all_dict["iteration"][max_idx] / min(all_dict['iteration']), 0),
                                 round(max_val * 100, 2),
                                 round(acc_G * 100, 2),
                                 round(acc_P * 100, 2),
                                 round(acc_V * 100, 2),
                                 round(acc_I * 100, 2),
                                 round(obj['iteration'] / min(all_dict['iteration']), 2),
                                 round(obj['eta_seconds'] / 3600, 1)))
                else:
                    all_txt = '[{}]'.format(case)
                    for i in range(len(all_dict[main_txt])):
                        all_txt += ' ({}){}'.format(round(all_dict["iteration"][i] / min(all_dict['iteration']), 0),
                                                    str(round(all_dict[main_txt][i] * 100, 2)).zfill(2))
                    print(all_txt)
            else:
                print('[{}] not exist'.format(case))
        else:
            print('[{}] not exist'.format(case))


