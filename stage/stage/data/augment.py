from ..utils import *

import cv2
import torchvision.transforms as transforms


# ##########################################
# # AUGMENTATION ###########################
# ##########################################


# def affine_random(image, ker=4):
#     old = np.float32([[120, 120], [136, 120], [128, 128]])
#     new = np.float32([augint(old[0], ker), 
#                       augint(old[1], ker),
#                       augint(old[2], 0)])
#     matrix = cv2.getAffineTransform(old, new)
#     image = cv2.warpAffine(image, matrix, image.shape[:2],
#                            borderMode=cv2.BORDER_REPLICATE)
    
#     return image


# def affine(image, num, kernel=(5, 5)):
#     if kernel == (0, 0) or kernel == 0:
#         return image
    
#     if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
#         raise ValueError("Only odd numbers are accepted for kernel size.")
#     if np.prod(kernel) <= num:
#         raise ValueError(f"Too large num({num}) for kernel({kernel}).")

#     decoder = np.zeros(kernel)
#     decoder = np.float32(list(zip(*np.where(decoder == 0)))) \
#             - np.float32(kernel) // 2
#     old = np.float32([[120, 120], [136, 120], [128, 128]])
#     new = np.vstack([old[:2] + decoder[num], old[2]])
#     matrix = cv2.getAffineTransform(old, new)
#     image = cv2.warpAffine(image, matrix, image.shape[:2],
#                            borderMode=cv2.BORDER_REPLICATE)
    
#     return image


# def adjust_contrast(image):
#     pass


def center_crop(image, size):
    try:
        size = int(size)
        size = (size, size)

    except TypeError:
        try:
            size = map(int, size)
            if len(list(size)) != 2:
                raise TypeError

        except TypeError:
            raise TypeError
    
    h, w, c = image.shape
    hn, wn = size
    if hn > h or wn > w:
        raise ValueError

    xlim = range((w - wn) // 2, (w + wn) // 2)
    ylim = range((h - hn) // 2, (h + hn) // 2)

    image = image[ylim, xlim]

    return image


def resize(image, size):
    if type(size) == int:
        size = (size, size)
    return cv2.resize(image, size)


# class RandomAugmenter(object):
#     def __init__(self, augs):
#         self._augs = self._set_augs(augs)
#         self._add_conversion(conv_dict)


#     def __call__(self, images):
#         if not self._augs:
#             return images
            
#         num_augs = self.transofrm
#         idx = np.random.randint(len(self) + 1)
#         if idx == len(self):
#             return images
#         return _augs[idx](images)


#     def __str__(self):
#         return self._augs


#     def __geitem__(self, idx):
#         return self._augs[idx]
        

#     def __len__(self):
#         return len(self._augs)


#     def _set_augs(self, augs):
#         transforms_dict = {
#             'center_crop':              transforms.CenterCrop,
#             'five_crop':                transforms.FiveCrop,
#             'ten_crop':                 transforms.TenCrop,
#             'random_crop':              transforms.RandomCrop,
#             'random_resized_crop':      transforms.RandomResizedCrop,
#             'random_rotation':          transforms.RandomRotation,
#             'random_affine':            transforms.RandomAffine,
#             'random_horizontal_filp':   transforms.RandomHorizontalFlip,
#             'random_vertical_flip':     transforms.RandomVerticalFlip,
#             'random_perspective':       transforms.RandomPerspective,
#             'resize':                   transforms.Resize,
#             'color_jitter':             transforms.ColorJitter,
#             'grayscale':                transforms.Grayscale,
#             'random_graysacle':         transforms.RandomGrayscale,
#             'pad':                      transforms.Pad,
#             'gaussian_blur':            transforms.GaussianBlur
#         }

#         aug_dict = {
#             'crop': center_crop,
#             'random_crop': random_crop,
#             'five_crop': five_crop,
#             'seg_crop': seg_crop,
#             'affine': affine,
#             'random_affine': affine_random,
#             'brightness': adjust_brightness,
#             'rotate': rotate,
#             'contrast': adjust_contrast,
#         }

#         aug_list = []
#         for aug in aug_dict.items():
#             try: aug_list.append(aug_dict[aug])
#             except: continue

#         return aug_list


#     def add_augs(self, *augs):
#         self._augs += augs


#     def get_augs(self):
#         return self._augs
