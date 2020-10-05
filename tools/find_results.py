import os
import json
from json import JSONDecoder, JSONDecodeError
import re

NOT_WHITESPACE = re.compile(r'[^\s]')


def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
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

folder_dir = '../logs'
folder_name = 'Sample2'
start_name = 'b30'
end_name = 'b31'

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

        for obj in decode_stacked(file.read()):
            if main_txt in obj.keys():
                print(obj)
            # print(obj)
        # jsondata = json.load(file.read())


