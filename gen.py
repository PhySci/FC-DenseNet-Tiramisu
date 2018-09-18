import os
from random import shuffle
import numpy as np
import cv2

class Fib:
    '''iterator that yields numbers in the Fibonacci sequence'''

    def __init__(self, img_pth=None, mask_pth=None, batch_size=1, shape=None, padding=None):
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
        self.train_files = os.listdir(img_pth)
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
        images = np.zeros([self.batch_size, self.shape[0], self.shape[1], 3])
        mask = np.zeros([self.batch_size, self.shape[0], self.shape[1]])

        for i, file in enumerate(file_list):
            img = cv2.imread(os.path.join(self.img_pth, file), -1)
            if self.padding is not None:
                img = cv2.copyMakeBorder(img, self.padding[0], self.padding[1], self.padding[2], self.padding[3],
                                         cv2.BORDER_REFLECT_101)
            images[i, :, :, :] = np.float32(img)/255.0
            if self.mask_pth is not None:
                img = cv2.imread(os.path.join(self.mask_pth, file), -1)
                if self.padding is not None:
                    img = cv2.copyMakeBorder(img, self.padding[0], self.padding[1], self.padding[2], self.padding[3],
                                             cv2.BORDER_REFLECT_101)
                mask[i, :, :] = np.int8(img/65535.0)

        if self.i >= self.max:
            raise StopIteration
        self.i +=1

        return images, mask, file_list