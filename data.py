import os
import glob
import pandas as pd
import shutil


path_to_data = 'data'
list_dir = os.listdir('data')
path_to_save = 'fall_data'

for dir in list_dir:
    path_img_dir = os.path.join(path_to_data, dir, 'rgb')
    path_to_csv = os.path.join(path_to_data, dir, 'labels.csv')
    
    df = pd.read_csv(path_to_csv)
    img_list = glob.glob(path_img_dir + '/*')
    
    for i, j in zip(df['index'], df['class']):
        if j == 3:
            path_to_img_pos = f'{path_img_dir}/{i}.png'
            img_len_pos = len(os.listdir(os.path.join(path_to_save, 'positive')))
            path_to_save_pos = os.path.join(path_to_save, 'positive') + f"/{img_len_pos}.png"
            shutil.copy(path_to_img_pos, path_to_save_pos)
        else:
            path_to_img_neg = f'{path_img_dir}/{i}.png'
            img_len_neg = len(os.listdir(os.path.join(path_to_save, 'negative')))
            path_to_save_neg = os.path.join(path_to_save, 'negative') + f"/{img_len_neg}.png"
            shutil.copy(path_to_img_neg, path_to_save_neg)
    
    print(f'[INFO] Completed {dir} Directory')
