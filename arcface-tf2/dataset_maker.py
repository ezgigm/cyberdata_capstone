import os
import shutil
import numpy as np

PATH = '../data/dataset_V3_original'

name_path_list = [os.path.join(os.path.join(PATH, folder_name), person_name) for folder_name in os.listdir(PATH) if not folder_name.startswith('.') for person_name in os.listdir(os.path.join(PATH, folder_name)) if not person_name.startswith('.')]

img_path_list = []
for name_path in name_path_list:
    img_path_list.extend([os.path.join(name_path, img_name) for img_name in os.listdir(name_path) if not img_name.startswith('.')])

#total 5503 images
dst = '/Users/1111613/Desktop/yujin/UM/699_Capstone/arcface-tf2/download_testset/v3_dataset'
i = 0
for src in img_path_list:
    li = src.split('/')
    li[-1] = li[-2] + '_' + li[-1]
    # shutil.copy(src, os.path.join(dst, li[-1]))

dst_list = sorted(os.listdir(dst))
issame = []
for i in range(len(dst_list)//2):
    file_1 = dst_list[2 * i]
    file_2 = dst_list[2 * i + 1]
    print(file_1, file_2)
    if file_1.split('_')[:2] == file_2.split('_')[:2]:
        issame.append(True)
        # np.append(issame, True)
    else:
        issame.append(False)
        # np.append(issame, False)
# print(len(issame), issame)
np.save(os.path.join(dst, 'v3_list.npy'), issame)