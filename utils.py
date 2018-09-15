from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import operator

import os

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time
def LOG(X, f=None):
	time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	if not f:
		print(time_stamp + " " + X)
	else:
		f.write(time_stamp + " " + X)

# Replaces a select value in an array with a new value
def replace_val_in_array(input_array, original_val, replace_val = sys.maxsize - 1):
    for index, item in enumerate(input_array):
        if item == original_val:
            input_array[index] = replace_val
    return input_array

def replaces_nan_in_array(input_array, replace_val=1.0):
    for index, item in enumerate(input_array):
        if math.isnan(item):
            input_array[index] = replace_val
    return input_array


# Compute the average segmentation accuracy across all classes
def compute_avg_accuracy(y_pred, y_true):
    # print(y_true.shape)
    w = y_true.shape[0]
    h = y_true.shape[1]
    total = w*h
    count = 0.0
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count = count + 1.0
    # print(count)
    return count / total

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(y_pred, y_true, num_classes=12):
    # print(y_true.shape)
    w = y_true.shape[0]
    h = y_true.shape[1]
    flat_image = np.reshape(y_true, w*h)
    total = []
    for val in range(num_classes):
        total.append((flat_image == val).sum())

    count = [0.0] * 12
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count[int(y_pred[i, j])] = count[int(y_pred[i, j])] + 1.0
    # print(count)

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('memory use:', memoryUse)


def one_hot_it(labels):
    """
    Какой же пиздец. Надо как-то это переварить и переписать
    :param labels:
    :return:
    """
    w = labels.shape[0]
    h = labels.shape[1]
    x = np.zeros([w,h,12])
    for i in range(0, w):
        for j in range(0, h):
            x[i,j,labels[i][j]]=1
    return x


def reverse_one_hot(image):
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,1])
    # print(image.shape)
    for i in range(0, w):
        for j in range(0, h):
            index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
            x[i, j] = index
    return x



def colour_dict(x):
    return {
        0: [128,128,128],
        1: [128,0,0],
        2: [192,192,128],
        3: [128,64,128],
        4: [60,40,222],
        5: [128,128,0],
        6: [192,128,128],
        7: [64,64,128],
        8: [64,0,128],
        9: [64,64,0],
        10: [0,128,192],
        11: [0,0,0]
    }[x]

def colour_code_segmentation(image):
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3])
    for i in range(0, w):
        for j in range(0, h):
            x[i, j, :] = colour_dict(image[i, j, 0])
    return x

# Colour codes:
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]