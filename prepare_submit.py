import os
import cv2
import numpy as np
import re

def my_rlc(arr):
    """

    :param arr:
    :return:
    """
    arr = np.insert(arr, 0, 0)
    arr = np.append(arr, 0)
    d = np.diff(np.int8(np.greater(arr, 0)))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return np.stack((starts, ends - starts), axis=1).flatten()

def main(pth=None):
    """
    Read and decode images in RLC format
    :param pth:
    :return:
    """
    s = ''
    with open("Output.txt", "w") as text_file:
        for file in os.listdir(pth):
            file_name, _ = os.path.splitext(file)
            img = cv2.imread(os.path.join(os.path.join(pth, file)), -1)
            s = np.array_str(my_rlc(img))
            print("{}, {}".format(file_name, re.sub(' +', ' ', s[1:-1])), file=text_file)


if __name__ == '__main__':
    main('../data/test/masks')