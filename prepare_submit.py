import os
import cv2
import numpy as np
import re
import datetime


def mask2rlc(arr):
    """

    :param arr:
    :return:
    """
    arr = arr.flatten(order='F')
    arr = np.insert(arr, 0, 0)
    arr = np.append(arr, 0)
    d = np.diff(np.int8(np.greater(arr, 0)))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    len = ends - starts
    assert (starts.shape == ends.shape)
    v = np.stack((starts+1, len), axis=1).flatten()
    s = np.array_str(v, max_line_width=99999)[1:-1]
    return s


def main(pth=None):
    """
    Read and decode images in RLC format
    :param pth:
    :return:
    """

    for file in os.listdir(pth):
        file_name, ext = os.path.splitext(file)
        if ext == '.png':
            try:
                img = cv2.imread(os.path.join(os.path.join(pth, file)), -1)
            except Exception as err:
                print(repr(err))
                print(file)
            s = mask2rlc(img)
            if s == '':
                print(file_name + ',', file=fid)
            else:
                print(file_name+','+re.sub(' +', ' ', s.strip()), file=fid)
        else:
            print(file)


if __name__ == '__main__':
    main('./predictions2')
