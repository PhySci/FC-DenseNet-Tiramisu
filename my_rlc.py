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
    return np.stack((starts, ends - starts), axis=1).flatten()


def main():
    """

    :return:
    """
    tests=[
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1]
    ]

    for test_arr in tests:
        v = my_rlc(np.array(test_arr))
        print(v)

if __name__ == '__main__':
    main()
