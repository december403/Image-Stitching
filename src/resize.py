from cv2 import cv2 
import os

source_file_path = './image/UAV_resized/'
files = os.listdir(source_file_path)
files = sorted(files)
file_num = len(files)
print(f'{source_file_path}{files[0]}')

for idx, file_name in enumerate(files):
    # new_file_name = file_name.replace('JPG','png')
    input = cv2.imread(f'{source_file_path}{file_name}')
    # pic = cv2.resize(input, (input.shape[1]//2, input.shape[0]//2), interpolation=cv2.INTER_CUBIC)
    h,w = input.shape[0:2]
    new_h = int(h*0.9)
    new_w = new_h
    h_diff = h - new_h
    w_diff = w - new_w
    pic = input[h_diff//2:h-h_diff//2, w_diff//2:w-w_diff//2]
    cv2.imwrite(f'{source_file_path}{idx}_crop.png', pic)