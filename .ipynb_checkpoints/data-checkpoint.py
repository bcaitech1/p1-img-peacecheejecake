from utils import *

import cv2

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms

from typing import Iterable, TypeVar

I = TypeVar('I', np.ndarray, list, int)



##########################################
# PREPROCESSING ##########################
##########################################


def preprocess(image: np.ndarray, size=256, segment=True, blur=True):
    # Normalize
    image = cv2.normalize(image, 0, 255)

    # Segmentation
    

    # Remove noise - Guassian blur
    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.bilateralFilter(image, 5, 75, 75)

    return image



##########################################
# SEGMENTATION ###########################
##########################################


def segment(image: np.ndarray):
    pass



##########################################
# AUGMENTATION ###########################
##########################################


def affine_random(image, ker=4):
    old = np.float32([[120, 120], [136, 120], [128, 128]])
    new = np.float32([augint(old[0], ker), 
                      augint(old[1], ker),
                      augint(old[2], 0)])
    matrix = cv2.getAffineTransform(old, new)
    image = cv2.warpAffine(image, matrix, image.shape[:2],
                           borderMode=cv2.BORDER_REPLICATE)
    
    return image


def affine(image, num, kernel=(5, 5)):
    if kernel == (0, 0) or kernel == 0:
        return image
    
    if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
        raise ValueError("Only odd numbers are accepted for kernel size.")
    if np.prod(kernel) <= num:
        raise ValueError(f"Too large num({num}) for kernel({kernel}).")

    decoder = np.zeros(kernel)
    decoder = np.float32(list(zip(*np.where(decoder == 0)))) \
            - np.float32(kernel) // 2
    old = np.float32([[120, 120], [136, 120], [128, 128]])
    new = np.vstack([old[:2] + decoder[num], old[2]])
    matrix = cv2.getAffineTransform(old, new)
    image = cv2.warpAffine(image, matrix, image.shape[:2],
                           borderMode=cv2.BORDER_REPLICATE)
    
    return image


def adjust_contrast(image):
    pass


def crop(image, size):
    height, width = image.shape[:2]
    x_start, x_end = (width - size) // 2, (width + size) // 2
    y_start, y_end = (height - size) // 2, (height + size) // 2
    image = image[x_start:x_end, y_start:y_end]
    image = cv2.resize(image, (height, width))
    image = image / 255.


class RandomAugmenter(object):
    def __init__(self, augs):
        self._augs = self._set_augs(augs)
        self._add_conversion(conv_dict)


    def __call__(self, images):
        if not self._augs:
            return images
            
        num_augs = self.transofrm
        idx = np.random.randint(len(self) + 1)
        if idx == len(self):
            return images
        return _augs[idx](images)


    def __str__(self):
        return self._augs


    def __geitem__(self, idx):
        return self._augs[idx]
        

    def __len__(self):
        return len(self._augs)


    def _set_augs(self, augs):
        transforms_dict = {
            'center_crop':              transforms.CenterCrop,
            'five_crop':                transforms.FiveCrop,
            'ten_crop':                 transforms.TenCrop,
            'random_crop':              transforms.RandomCrop,
            'random_resized_crop':      transforms.RandomResizedCrop,
            'random_rotation':          transforms.RandomRotation,
            'random_affine':            transforms.RandomAffine,
            'random_horizontal_filp':   transforms.RandomHorizontalFlip,
            'random_vertical_flip':     transforms.RandomVerticalFlip,
            'random_perspective':       transforms.RandomPerspective,
            'resize':                   transforms.Resize,
            'color_jitter':             transforms.ColorJitter,
            'grayscale':                transforms.Grayscale,
            'random_graysacle':         transforms.RandomGrayscale,
            'pad':                      transforms.Pad,
            'gaussian_blur':            transforms.GaussianBlur
        }

        aug_dict = {
            'crop': center_crop,
            'random_crop': random_crop,
            'five_crop': five_crop,
            'seg_crop': seg_crop,
            'affine': affine,
            'random_affine': affine_random,
            'brightness': adjust_brightness,
            'rotate': rotate,
            'contrast': adjust_contrast,
        }

        aug_list = []
        for aug in aug_dict.items():
            try: aug_list.append(aug_dict[aug])
            except: continue

        return aug_list


    def add_augs(self, *augs):
        self._augs += augs


    def get_augs(self):
        return self._augs



##########################################
# DATASET ################################
##########################################   
        

class BasicDataset(Dataset):
    def __init__(self, data: np.ndarray, mode: str, preprocess: bool=True, augment: bool=False):
        self.data = data
        if mode == 'train':
            self.data = self.balancing(data)
            
        self.mode = mode
        self.preprocess = preprocess
        self.augment = augment
        
        
    def __getitem__(self, idx: I):
        image_file = self.data[idx]
        image = cv2.imread(image_file)
        
        # debug
        if image is None:
            print(image_file)
            
        h, w, _ = image.shape
        image = image[h-w:, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment: image = affine_random(image, ker=1)
        if self.preprocess: image = preprocess(image)
        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)
        
        image_name = image_file.split('/')[-1]
        if self.mode in ('train', 'valid'):
            label = int(image_name[:2])
            return image, label
        else:
            return image, image_name
    
    
    def __len__(self):
        return len(self.data)
    
    
    def balancing(self, data):
        class_counter = [0] * 18
        for filepath in self.data:
            label = int(filepath.split('/')[-1][:2])
            class_counter[label] += 1
        max_num_class = max(class_counter)
        
        data = []
        for label in range(18):
            data_in_label = [filepath for filepath in self.data if int(filepath.split('/')[-1][:2]) == label]
            num_class = len(data_in_label)
            if num_class < max_num_class:
                data_in_label = np.array(data_in_label)
                random_idx = np.random.randint(0, num_class, max_num_class)
                data_in_label = list(data_in_label[random_idx])

            data += data_in_label
            
        return data
        
        
class AugDataset(BasicDataset):
    def __init__(self, root: os.PathLike, mode: str, num: int, 
                 affine_kernel=(5, 5), gaussian_kernel=(3, 3)):
        super(AugDataset, self).__init__(root, mode)
        self.aug_num = num
        self.affine_kernel = affine_kernel
        self.gaussian_kernel = gaussian_kernel


    def __getitem__(self, idx: I):
        image_file = self.data[idx]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = affine(image, self.aug_num, self.affine_kernel)
        image = image / 255.
        iamge = cv2.normalize(image, 0, 255)
        image = cv2.GaussianBlur(image, self.gaussian_kernel, 0)
        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)
        
        image_name = image_file.split('/')[-1]
        label = int(image_name[:3])

        return image, label



def train_valid_split(dataset: Dataset, valid_ratio: float=0.2, shuffle: bool=True):
    data_size = len(dataset)
    valid_size = int(data_size * valid_ratio)
    
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    indices_train, indices_valid = indices[valid_size:], indices[:valid_size]
    train, valid = Subset(dataset, indices_train), Subset(dataset, indices_valid)

    return train, valid


def train_valid_raw_split(root: os.PathLike, valid_ratio: float=0.2, shuffle: bool=True):
    raw_data = glob(os.path.join(root, "*.jpg"))
    raw_data = np.array(raw_data)
    
    data_size = len(raw_data)
    valid_size = int(data_size * valid_ratio)
    
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    
    indices_train, indices_valid = indices[valid_size:], indices[:valid_size]
    train, valid = raw_data[indices_train], raw_data[indices_valid]

    return train, valid