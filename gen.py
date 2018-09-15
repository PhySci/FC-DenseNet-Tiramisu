import os
from random import shuffle
import numpy as np
import cv2

class Fib:
    '''iterator that yields numbers in the Fibonacci sequence'''

    def __init__(self, img_pth=None, mask_pth=None, batch_size=1, shape=None):
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
        images = None
        mask = None

        for i, file in enumerate(file_list):
            arr = np.float32(cv2.imread(os.path.join(self.img_pth, file), -1))[:self.shape[0], :self.shape[1], :]/ 255.0
            if images is None:
                images = np.zeros([self.batch_size, arr.shape[0], arr.shape[1], arr.shape[2]])
            images[i, :, :, :] = arr

            if self.mask_pth is not None:
                arr = np.int8(cv2.imread(os.path.join(self.mask_pth, file), -1)[:self.shape[0], :self.shape[1]])
                if mask is None:
                    mask = np.zeros([self.batch_size, arr.shape[0], arr.shape[1], 12], dtype=np.int8)
                mask[i, :, :, :] = one_hot_it(arr)

        if self.i >= self.max:
            raise StopIteration
        self.i +=1

        return images, mask

def one_hot_it(labels):
    """
    Какой же пиздец. Надо как-то это переварить и переписать
    :param labels:
    :return:
    """
    w = labels.shape[0]
    h = labels.shape[1]
    x = np.zeros([w, h, 12])
    for i in range(0, w):
        for j in range(0, h):
            x[i,j,labels[i][j]]=1
    return x


if __name__ == '__main__':
    print('S')

    for img, msk in Fib(img_pth='./CamVid/train', mask_pth='./CamVid/train_labels'):
        print(img)
        print(msk)