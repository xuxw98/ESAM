import os
import os.path as osp
import numpy as np
import natsort

txt_file = 'scannetv2_val.txt'
lines = open(txt_file).readlines()
lines = [line.strip() for line in lines]
lines = natsort.natsorted(lines)

# write sorted lines to a new file
new_txt_file = 'scannetv2_val_sorted.txt'
with open(new_txt_file, 'w') as f:
    for line in lines:
        f.write(line + '\n')
print(f'File {new_txt_file} has been written.')