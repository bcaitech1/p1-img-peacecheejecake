import os
import pandas as pd
import shutil
from glob import glob


def gender_dict(x: str):
    return int(x == 'female')


def age_dict(x: int):
    if x < 30: return 0
    elif x < 57: return 1
    elif x >= 60: return 2


def mask_dict(filename: str):
    if 'incorrect' in filename:
        return 1
    elif 'normal' in filename:
        return 2
    else:
        return 0


train_path = '/opt/ml/input/data/train'
train_image_path = 'opt/ml/input/data/train/_all'
train_origin_path = 'opt/ml/input/data/train/images'


info = pd.read_csv(os.path.join(train_path, 'train.csv'))
info = info.drop()
info['class_num'] = 3 * info.gender.map(gender_dict) + info.age.map(age_dict)
info = info.drop(['gender', 'age', 'race'], axis=1)


# image rearangement
train_size = len(info) * 7

dest_base = train_image_path
if not os.path.exists(dest_base):
    os.mkdir(dest_base)
else:
    shutil.rmtree(dest_base)
    os.mkdir(dest_base)
        
# elif len(os.listdir(dest_base)) < train_size:
#     shutil.rmtree(dest_base)
#     os.mkdir(dest_base)
# else:
#     raise('Dest path already exists.')


# label and copy images

print(f"Start copying: {train_origin_path} -> {train_image_path}")

for idx in info.index:
    id, path, class_num = info.loc[idx]
    origin_base = os.path.join(train_origin_path, path)
    for image_num, origin in enumerate(glob(os.path.join(origin_base, '*'))):
        print(f'\rCopying {idx * 7 + image_num + 1}/{train_size}', end='')

        origin_name, ext = origin.split('/')[-1].split('.')
        dest_name = f'{6 * mask_dict(origin_name) + class_num:02d}_{path}_{image_num}.jpg'
        dest = os.path.join(dest_base, dest_name)
        shutil.copy(origin, dest)
else:
    copied_num = len(glob(os.path.join(train_image_path, '*')))
    print(f'\nCopied {copied_num} images.')