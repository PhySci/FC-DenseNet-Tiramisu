import cv2
import os
from tqdm import tqdm
import numpy as np

from albumentations import (
    VerticalFlip,
    Compose,
    Transpose,
    RandomRotate90,
)

def main():

    img_pth = './soil/train/images'
    mask_pth = './soil/train/masks'
    aug_img_pth = './soil/augment/images'
    aug_mask_pth = './soil/augment/masks'

    aug = Compose([VerticalFlip(p=0.8),
                   RandomRotate90(p=0.8),
                   Transpose(p=0.8),
                   RandomRotate90(p=1.0)])

    file_list = os.listdir(img_pth)
    i = 0
    for file in tqdm(file_list):
        image = cv2.imread(os.path.join(img_pth, file))
        mask = cv2.imread(os.path.join(mask_pth, file), 0)

        for i in range(5):

            if i > 0:
                if np.count_nonzero(image) == 0:
                    i += 1
                    continue

                augmented = aug(image=image, mask=mask)
                cv2.imwrite(os.path.join(aug_img_pth, 'aug'+str(i)+'_'+file),  augmented['image'])
                cv2.imwrite(os.path.join(aug_mask_pth, 'aug'+str(i)+'_'+file),  augmented['mask'])
            else:

                cv2.imwrite(os.path.join(aug_img_pth, 'aug' + str(i) + '_' + file), image)
                cv2.imwrite(os.path.join(aug_mask_pth, 'aug' + str(i) + '_' + file), mask)

    print(i)

if __name__ == '__main__':
    main()