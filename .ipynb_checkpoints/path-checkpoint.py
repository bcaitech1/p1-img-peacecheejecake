import os

data_path = '/opt/ml/input/data'

train_path = os.path.join(data_path, 'train')
train_origin_path = os.path.join(train_path, 'images')
train_image_path = os.path.join(train_origin_path, '_all')

eval_path = os.path.join(data_path, 'eval')
eval_image_path = os.path.join(eval_path, 'images')
