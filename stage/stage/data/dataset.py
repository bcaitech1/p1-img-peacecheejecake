# from torch._C import int16
from ..utils import *
from .augment import *
from .preprocess import preprocess
from .functional import *

from collections import Counter
import cv2
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import albumentations as A

from typing import TypeVar, Iterable


##########################################
# DATASET ################################
##########################################   
        

class BasicDataset(Dataset):
    NUM_CLASSES = 18

    def __init__(
        self, 
        data: Iterable, 
        labeled: bool, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None
    ):
        super(BasicDataset, self).__init__()
        self.data = data
        self.labeled = labeled
        self.preprocess = preprocess
        self.crop_size = crop_size
        self.resize = resize
        self.num_classes = 18
        
        
    def __getitem__(self, idx: int):
        image_file = self.data[idx]
        image = cv2.imread(image_file)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.preprocess: 
            image = preprocess(image)
        else:
            image = image / 255.
        if self.crop_size:
            image = center_crop(image, self.crop_size)
        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)
        
        image_name = image_file.split('/')[-1]
        if self.labeled:
            label = int(image_name[:2])
            return image, label
        else:
            return image, image_name
    
    
    def __len__(self):
        return len(self.data)
    
    
    # def upscale(self, data):
    #     class_counter = [0] * 18
    #     for filepath in self.data:
    #         label = int(filepath.split('/')[-1][:2])
    #         class_counter[label] += 1
    #     max_num_class = max(class_counter)
        
    #     data = []
    #     for label in range(18):
    #         data_in_label = [filepath for filepath in self.data if int(filepath.split('/')[-1][:2]) == label]
    #         num_class = len(data_in_label)
    #         if num_class < max_num_class:
    #             data_in_label = np.array(data_in_label)
    #             random_idx = np.random.randint(0, num_class, max_num_class)
    #             data_in_label = list(data_in_label[random_idx])

    #         data += data_in_label
            
    #     return data
        

class AgeDataset(BasicDataset):
    '''
    0 <- under 30
    1 <- between 30 and 60
    2 <- over 60
    '''
    NUM_CLASSES = 3

    def __init__(
        self, 
        data: Iterable, 
        labeled: bool, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None
    ):
        super(AgeDataset, self).__init__(data, labeled, preprocess, crop_size, resize)
        self.num_classes = 3


    def __getitem__(self, idx: int):
        image_file = self.data[idx]
        image = cv2.imread(image_file)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.preprocess: 
            image = preprocess(image)
        else:
            image = image / 255.
        if self.crop_size:
            image = center_crop(image, self.crop_size)
        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)
        
        image_name = image_file.split('/')[-1]
        if self.labeled:
            label = int(image_name[:2])
            return image, age_from_label(label)
        else:
            return image, image_name


class GenderDataset(BasicDataset):
    '''
    0 <- male
    1 <- female
    '''
    NUM_CLASSES = 2

    def __init__(
        self, 
        data: Iterable, 
        labeled: bool, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None
    ):
        super(GenderDataset, self).__init__(data, labeled, preprocess, crop_size, resize)
        self.num_classes = 2


    def __getitem__(self, idx: int):
        image_file = self.data[idx]
        image = cv2.imread(image_file)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.preprocess: 
            image = preprocess(image)
        else:
            image = image / 255.
        if self.crop_size:
            image = center_crop(image, self.crop_size)
        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)
        
        image_name = image_file.split('/')[-1]
        if self.labeled:
            label = int(image_name[:2])
            return image, gender_from_label(label)
        else:
            return image, image_name


class MaskDataset(BasicDataset):
    '''
    0 <- wear
    1 <- incorrect
    2 <- not wear
    '''
    NUM_CLASSES = 3

    def __init__(
        self, 
        data: Iterable, 
        labeled: bool, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None
    ):
        super(MaskDataset, self).__init__(data, labeled, preprocess, crop_size, resize)
        self.num_classes = 3


    def __getitem__(self, idx: int):
        image_file = self.data[idx]
        image = cv2.imread(image_file)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.preprocess: 
            image = preprocess(image)
        else:
            image = image / 255.
        if self.crop_size:
            image = center_crop(image, self.crop_size)
        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)
        
        image_name = image_file.split('/')[-1]
        if self.labeled:
            label = int(image_name[:2])
            return image, mask_from_label(label)
        else:
            return image, image_name


class DefaultUpscaleAugments:
    augments = [
        A.HorizontalFlip(always_apply=True),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, always_apply=True),
        A.RandomBrightnessContrast(always_apply=True)
    ]
            

class UpscaledDataset(BasicDataset):
    def __init__(
        self,
        data: Iterable, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None,
        augments: Iterable = None,
    ):
        """Need to add parameter :labeled: to implement TTA."""
        super(UpscaledDataset, self).__init__(data, True, preprocess, crop_size, resize)
        
        if augments:
            self.augments = augments
        else:
            self.augments = DefaultUpscaleAugments.augments

        self.original_size = len(self.data)
        self.upscale_data_with_random_aug()
        self.num_classes = 18


    def __getitem__(self, idx):
        image_file, augment_idx = self.data[idx]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = self.augments[augment_idx](image=image)['image']
        except IndexError:
            pass

        if self.preprocess:
            image = preprocess(image)
        else:
            image = image / 255.

        if self.crop_size:
            image = center_crop(image, self.crop_size)

        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)

        image_name = image_file.split('/')[-1]
        label = int(image_name[:2])

        return image, label
        

    def upscale_data_with_random_aug(self):
        upscaled_data = []
        
        num_augments = len(self.augments) + 1
        num_classes = self.__class__.NUM_CLASSES

        labels_on_data = [int(path.split('/')[-1][:2]) for path in self.data]
        class_counter = Counter(labels_on_data)
        max_class, max_count = class_counter.most_common(1)[0]

        for class_idx in range(num_classes):
            data_in_class = np.array([path for path in self.data if int(path.split('/')[-1][:2]) == class_idx])
            random_indices = np.random.randint(0, class_counter[class_idx], max_count)
            random_augments = np.random.randint(0, num_augments + 1, max_count)
            upscaled_data += zip(data_in_class[random_indices], random_augments)
        
        self.data = upscaled_data


    def augmentations(self):
        return self.augments


    def __str__(self):
        return f"Dataset upscaled with random augmentations. {self.original_size} -> {len(self)}"


class UpscaledAgeDataset(UpscaledDataset):
    def __init__(
        self,
        data: Iterable, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None,
        augments: Iterable = None,
    ):
        super(UpscaledAgeDataset, self).__init__(data, preprocess, crop_size, resize, augments)
        self.num_classes = 3
        

    def __getitem__(self, idx):
        image_file, augment_idx = self.data[idx]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = self.augments[augment_idx](image=image)['image']
        except IndexError:
            pass

        if self.preprocess:
            image = preprocess(image)
        else:
            image = image / 255.

        if self.crop_size:
            image = center_crop(image, self.crop_size)

        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)

        image_name = image_file.split('/')[-1]
        label = int(image_name[:2])
        label = age_from_label(label)

        return image, label


    def upscale_data_with_random_aug(self):
        upscaled_data = []
        
        num_augments = len(self.augments) + 1
        num_classes = self.__class__.NUM_CLASSES

        label = lambda x: age_from_label(int(x.split('/')[-1][:2]))
        labels_on_data = [label(path) for path in self.data]
        class_counter = Counter(labels_on_data)
        max_class, max_count = class_counter.most_common(1)[0]

        for class_idx in range(num_classes):
            data_in_class = np.array([path for path in self.data if label(path) == class_idx])
            random_indices = np.random.randint(0, class_counter[class_idx], max_count)
            random_augments = np.random.randint(0, num_augments + 1, max_count)
            upscaled_data += zip(data_in_class[random_indices], random_augments)
        
        self.data = upscaled_data

        
    def __str__(self):
        return f"Age dataset upscaled with random augmentations. {self.original_size} -> {len(self)}"


class UpscaledGenderDataset(UpscaledDataset):
    def __init__(
        self,
        data: Iterable, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None,
        augments: Iterable = None,
    ):
        super(UpscaledGenderDataset, self).__init__(data, preprocess, crop_size, resize, augments)
        self.num_classes = 2

    def __getitem__(self, idx):
        image_file, augment_idx = self.data[idx]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = self.augments[augment_idx](image=image)['image']
        except IndexError:
            pass

        if self.preprocess:
            image = preprocess(image)
        else:
            image = image / 255.

        if self.crop_size:
            image = center_crop(image, self.crop_size)

        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)

        image_name = image_file.split('/')[-1]
        label = int(image_name[:2])
        label = gender_from_label(label)

        return image, label


    def upscale_data_with_random_aug(self):
        upscaled_data = []
        
        num_augments = len(self.augments) + 1
        num_classes = self.__class__.NUM_CLASSES

        label = lambda x: gender_from_label(int(x.split('/')[-1][:2]))
        labels_on_data = [label(path) for path in self.data]
        class_counter = Counter(labels_on_data)
        max_class, max_count = class_counter.most_common(1)[0]

        for class_idx in range(num_classes):
            data_in_class = np.array([path for path in self.data if label(path) == class_idx])
            random_indices = np.random.randint(0, class_counter[class_idx], max_count)
            random_augments = np.random.randint(0, num_augments + 1, max_count)
            upscaled_data += zip(data_in_class[random_indices], random_augments)
        
        self.data = upscaled_data


    def __str__(self):
        return f"Gender dataset upscaled with random augmentations. {self.original_size} -> {len(self)}"


class UpscaledMaskDataset(UpscaledDataset):
    def __init__(
        self,
        data: Iterable, 
        preprocess: bool = False, 
        crop_size: Iterable[int] = None,
        resize: Iterable[int] = None,
        augments: Iterable = None,
    ):
        super(UpscaledMaskDataset, self).__init__(data, preprocess, crop_size, resize, augments)
        self.num_classes = 3
        

    def __getitem__(self, idx):
        image_file, augment_idx = self.data[idx]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            image = self.augments[augment_idx](image=image)['image']
        except IndexError:
            pass

        if self.preprocess:
            image = preprocess(image)
        else:
            image = image / 255.

        if self.crop_size:
            image = center_crop(image, self.crop_size)

        if self.resize:
            image = resize(image, self.resize)

        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)

        image_name = image_file.split('/')[-1]
        label = int(image_name[:2])
        label = mask_from_label(label)
        
        return image, label


    def upscale_data_with_random_aug(self):
        upscaled_data = []
        
        num_augments = len(self.augments) + 1
        num_classes = self.__class__.NUM_CLASSES

        label = lambda x: mask_from_label(int(x.split('/')[-1][:2]))
        labels_on_data = [label(path) for path in self.data]
        class_counter = Counter(labels_on_data)
        max_class, max_count = class_counter.most_common(1)[0]

        for class_idx in range(num_classes):
            data_in_class = np.array([path for path in self.data if label(path) == class_idx])
            random_indices = np.random.randint(0, class_counter[class_idx], max_count)
            random_augments = np.random.randint(0, num_augments + 1, max_count)
            upscaled_data += zip(data_in_class[random_indices], random_augments)
        
        self.data = upscaled_data


    def __str__(self):
        return f"Mask dataset upscaled with random augmentations. {self.original_size} -> {len(self)}"



def train_valid_split(
    dataset: Dataset, 
    valid_ratio: float = 0.2, 
    shuffle: bool = True
):    
    data_size = len(dataset)
    valid_size = int(data_size * valid_ratio)
    
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    
    indices_train, indices_valid = indices[valid_size:], indices[:valid_size]
    train, valid = Subset(dataset, indices_train), Subset(dataset, indices_valid)

    return train, valid


def train_valid_raw_split(
    data: Iterable, 
    valid_ratio: float = 0.2, 
    shuffle: bool = True
):
    data_size = len(data)
    valid_size = int(data_size * valid_ratio)
    
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    
    indices_train, indices_valid = indices[valid_size:], indices[:valid_size]

    data = np.array(data)
    train, valid = data[indices_train], data[indices_valid]

    return train, valid


def age_from_label(label: int):
    '''
    0 <- under 30
    1 <- between 30 and 60
    2 <- over 60
    '''
    return label % 3


def gender_from_label(label: int):
    '''
    0 <- male
    1 <- female
    '''
    return (label // 3) % 2


def mask_from_label(label: int):
    '''
    0 <- wear
    1 <- incorrect
    2 <- not wear
    '''
    return label // 6


def labels(ages: pd.Series, genders: pd.Series, masks: pd.Series):
    return ages + 3 * genders + 6 * masks
    


class NormalAgeDataset(BasicDataset):
    '''
    Re-labeled ground truth of age prediction.
    '''

    info_file = '/opt/ml/input/data/train/train.csv'

    def __init__(self, data, upscaled: bool = False):
        super(NormalAgeDataset, self).__init__(data, labeled=True)

        self.data = data
        # self.preprocess = preprocess
        self.upscaled = upscaled
        self.train_info = pd.read_csv(NormalAgeDataset.info_file)

        if upscaled:
            self.augments = DefaultUpscaleAugments.augments


    def __getitem__(self, idx: int):
        if self.upscaled:
            image_path, aug_idx = self.data[idx]
        else:
            image_path = self.data[idx]  

        image_folder = '_'.join(image_path.split('/')[-1].split('_')[1:5]).split('.')[0]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.upscaled:
            image = self.augments[aug_idx](image=image)['image']
        image = image / 255.
        image = torch.tensor(image).permute((2, 0, 1)).to(torch.float)

        hard_label = self.train_info.loc[self.train_info.path == image_folder].age.iloc[0]

        soft_label = torch.zeros(3).to(torch.float)
        soft_label[0] = age_norm(hard_label, mu=15, sigma=15) if hard_label >= 15 else 1.
        soft_label[1] = age_norm(hard_label, mu=45, sigma=15)
        soft_label[2] = age_norm(hard_label, mu=60, sigma=5) if hard_label < 60 else 1.

        return image, soft_label



class RandomCutMixTrainDataset(BasicDataset):
    NUM_CLASSES = 18
    def __init__(self, data, kernel_size: int = 256, num_classes: int = 18):
        self.data = data
        self.kernel_size = kernel_size
        self.num_classes = num_classes


    def __getitem__(self, idx: int):
        '''
        image1 -> background
        image2 -> in kernel(randomly paired with image1)
        '''
        image1 = self.data[idx]
        data_size = len(self.data)

        for image1 in self.data:
            image2 = self.data[np.random.randint(data_size)]
            while image2 == image1:
                image2 = self.data[np.random.randint(data_size)]

        y1 = int(image1.split('/')[-1][:2])
        y2 = int(image2.split('/')[-1][:2])

        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image_mixed = image1[:, :]

        # Assume that the sizes of image1 and image2 are the same.
        h, w, _ = image1.shape
        if self.kernel_size > w:
            print("WARNING: kernel size was shrinked to width of image.")
            self.kernel_size = w

        ys, xs = np.random.randint(h - self.kernel_size), np.random.randint(w - self.kernel_size)
        ye, xe = ys + self.kernel_size, xs + self.kernel_size

        image_mixed[ys:ye, xs:xe] = image2[ys:ye, xs:xe]
        image_mixed = cv2.cvtColor(image_mixed, cv2.COLOR_BGR2RGB)
        image_mixed = image_mixed / 255.
        image_mixed = torch.tensor(image_mixed, dtype=torch.float).permute((2, 0, 1))

        # Random lambda from uniform distribution [0, 1]
        lambda_ = np.random.randint(0, 1)
        y_mixed = y1 * lambda_ + y2 * (1 - lambda_)

        return image_mixed, y_mixed


    def __len__(self):
        return len(self.data)


class CenterCutMixTrainDataset(BasicDataset):
    def __init__(self, data, kernel_size: int = 256, num_classes: int = 18):
        self.data = data
        self.kernel_size = kernel_size
        self.num_classes = num_classes


    def __getitem__(self, idx: int):
        '''
        image1 -> background
        image2 -> in kernel(randomly paired with image1)
        '''
        image1 = self.data[idx]
        data_size = len(self.data)

        for image1 in self.data:
            image2 = self.data[np.random.randint(data_size)]
            while image2 == image1:
                image2 = self.data[np.random.randint(data_size)]

        y1 = int(image1.split('/')[-1][:2])
        y2 = int(image2.split('/')[-1][:2])

        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image_mixed = image1[:, :]

        # Assume that the sizes of image1 and image2 are the same.
        h, w, _ = image1.shape
        if self.kernel_size > w:
            print("WARNING: kernel size was shrinked to width of image.")
            self.kernel_size = w

        ys, xs = (h - self.kernel_size) // 2, (w - self.kernel_size) // 2
        ye, xe = ys + self.kernel_size, xs + self.kernel_size

        image_mixed[ys:ye, xs:xe] = image2[ys:ye, xs:xe]
        image_mixed = cv2.cvtColor(image_mixed, cv2.COLOR_BGR2RGB)
        image_mixed = image_mixed / 255.
        image_mixed = torch.tensor(image_mixed, dtype=torch.float).permute((2, 0, 1))

        # Random lambda from uniform distribution [0, 1]
        lambda_ = np.random.randint(0, 1)
        y_mixed = y1 * lambda_ + y2 * (1 - lambda_)

        return image_mixed, y_mixed


    def __len__(self):
        return len(self.data)


# class RandomCutMixTestDataset(BasicDataset):
#     NUM_CLASSES = 18
#     def __init__(self, data, kernel_size: int = 256, num_classes: int = 18):
#         self.data = data
#         self.kernel_size = kernel_size
#         RandomCutMixTestDataset.NUM_CLASSES = num_classes ##### 반드시 고치기


#     def __getitem__(self, idx: int):
#         '''
#         image1 -> background
#         image2 -> in kernel(randomly paired with image1)
#         '''
#         image1_name = self.data[idx]
#         data_size = len(self.data)

#         for image1 in self.data:
#             image2_name = self.data[np.random.randint(data_size)]
#             while image2_name == image1_name:
#                 image2_name = self.data[np.random.randint(data_size)]

#         image1 = cv2.imread(image1_name)
#         image2 = cv2.imread(image2_name)
#         image_mixed = image1[:, :]

#         # Assume that the sizes of image1 and image2 are the same.
#         h, w, _ = image1.shape
#         if self.kernel_size > w:
#             print("WARNING: kernel size was shrinked to width of image.")
#             self.kernel_size = w

#         ys, xs = np.random.randint(h - self.kernel_size), np.random.randint(w - self.kernel_size)
#         ye, xe = ys + self.kernel_size, xs + self.kernel_size

#         image_mixed[ys:ye, xs:xe] = image2[ys:ye, xs:xe]
#         image_mixed = cv2.cvtColor(image_mixed, cv2.COLOR_BGR2RGB)
#         image_mixed = image_mixed / 255.
#         image_mixed = torch.tensor(image_mixed, dtype=torch.float).permute((2, 0, 1))

#         # Random lambda from uniform distribution [0, 1]
#         lambda_ = np.random.randint(0, 1)

#         return image_mixed, int(image1_name.split('/')[-1][:2])


    # def __len__(self):
    #     return len(self.data)


class SimpleTTA:
    augments = [
        A.Rotate(limit=15, always_apply=True),
        A.HorizontalFlip(always_apply=True),
        A.ColorJitter(always_apply=True),
        A.Blur(blur_limit=3, always_apply=True),
        A.GaussNoise(var_limit=20, always_apply=True),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, always_apply=True)
    ]
    

class SimpleTTADataset(Dataset):
    def __init__(self, data, augment):
        super(SimpleTTADataset, self).__init__()
        self.augment = augment
        self.data = data
    

    def __getitem__(self, idx: int):
        image_path = self.data[idx]
        image_file = image_path.split('/')[-1]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augment(image=image)['image']
        image = image / 255.
        image = torch.tensor(image).permute(2, 0, 1).to(torch.float)

        return image, image_file


    def __len__(self):
        return len(self.data)


