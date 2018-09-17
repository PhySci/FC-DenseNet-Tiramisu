import numpy as np

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
    return np.stack((starts+1, ends - starts), axis=1).flatten()

def mask_to_rle(mask):
    mask_flat = mask.flatten('F')
    flag = 0
    rle_list = list()
    for i in range(mask_flat.shape[0]):
        if flag == 0:
            if mask_flat[i] == 1:
                flag = 1
                starts = i+1
                rle_list.append(starts)
        else:
            if mask_flat[i] == 0:
                flag = 0
                ends = i
                rle_list.append(ends-starts+1)
    if flag == 1:
        ends = mask_flat.shape[0]
        rle_list.append(ends-starts+1)
    #sanity check
    if len(rle_list) % 2 != 0:
        print('NG')
    if len(rle_list) == 0:
        rle = np.nan
    else:
        rle = ' '.join(map(str,rle_list))
    return rle

def main():
    """

    :return:
    """
    tests=[
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0]
    ]
    test = np.array(tests)
    v1 = my_rlc(test.flatten(order='F'))
    print(v1)

    v2 = mask_to_rle(test)
    print(v2)

if __name__ == '__main__':
    main()
