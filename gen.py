import os
from random import shuffle
import numpy as np
import cv2

class Fib:
    """
    Image and mask generator
    """

    def __init__(self, img_pth=None, mask_pth=None, batch_size=1, shape=None, padding=None, flip=False):
        """
        Constructor
        :param img_pth:
        :param mask_pth:
        :param max:
        """
        self.img_pth = img_pth
        self.mask_pth = mask_pth
        self.batch_size = batch_size
        self.i =0
        self.shape = shape
        self.padding = padding
        self.flip = flip
        file_list = os.listdir(img_pth)
        shuffle(file_list)
        self.train_files = file_list

        self.max = np.ceil(len(self.train_files)/self.batch_size)

    def __iter__(self):
        """

        :return:
        """
        return self

    def __next__(self):
        """

        :return:
        """
        start = self.i*self.batch_size
        end = (self.i+1)*self.batch_size
        file_list = self.train_files[start:end]
        images = np.zeros([self.batch_size, self.shape[0], self.shape[1], 1])
        mask = np.zeros([self.batch_size, self.shape[0], self.shape[1]])

        for i, file in enumerate(file_list):
            img = cv2.imread(os.path.join(self.img_pth, file), 0)
            if self.padding is not None:
                img = cv2.copyMakeBorder(img, self.padding[0], self.padding[1], self.padding[2], self.padding[3],
                                         cv2.BORDER_REFLECT_101)
            images[i, :, :, 0] = img[:,:]/255.0
            
            if self.mask_pth is not None:
                img = cv2.imread(os.path.join(self.mask_pth, file), 0)
                if self.padding is not None:
                    img = cv2.copyMakeBorder(img, self.padding[0], self.padding[1], self.padding[2], self.padding[3],
                                             cv2.BORDER_REFLECT_101)
                img = np.int8(img/255.0)
                mask[i, :, :] = img

        if self.i >= self.max:
            raise StopIteration
        self.i +=1

        return images[:, :, :, 0:1], mask, file_list
