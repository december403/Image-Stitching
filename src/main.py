import argparse
import os
import subprocess



def stitch(source_file_path,destination_file_path):
    files = os.listdir(source_file_path)
    files = sorted(files)
    file_num = len(files)
    if file_num == 1:
        return True
    print(files)
    last_output = None
    for i in range(file_num//2):
        tar_img = source_file_path + files.pop(0)
        ref_img = source_file_path + files.pop(0)
        subprocess.run(['python3', './src/main_APAP.py',  tar_img, ref_img])
        subprocess.run(['python3', './src/main_seam_cutting.py',  f'{destination_file_path}result{i:02d}.png'])
        print(f'progress:{i:02d}')
        last_output =  f'{destination_file_path}result{i:02d}.png'

    if len(files)>0:
        print(f'level:final')
        tar_img = source_file_path + files.pop(0)
        ref_img = last_output
        subprocess.run(['python3', './src/main_APAP.py',  tar_img, ref_img])
        subprocess.run(['python3', './src/main_seam_cutting.py',  f'{destination_file_path}result{i:02d}.png'])
    return False

import time
start_idx = 0
start = time.time()
while True:
    print(f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++current idx: {start_idx}')
    source_file_path = f'./image/level{start_idx}/'
    destination_file_path = f'./image/level{start_idx+1}/'

    isFinished = stitch(source_file_path,destination_file_path)

    if isFinished:
        break
    else:
        start_idx += 1

print(f'takes {time.time()-start}')