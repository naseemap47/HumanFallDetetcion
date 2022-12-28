import random
import glob
import shutil


img_list = glob.glob('fall_data/negative/*')
random.shuffle(img_list)
new_list = img_list[:int(len(img_list)/2)]
for img in new_list:
    shutil.copy(img, 'fall_data/neg')
