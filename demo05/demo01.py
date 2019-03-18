# numpy 的基本操作

import numpy as np


# ndim size shape itemsize 属性
def base1():
    # numpy 版本号：1.16.2
    # 有时候做深度学习会遇到版本不兼容的情况
    print("numpy version is", np.__version__)

    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    # ndim 表示维度 就是有几个括号
    print("arr1.ndim:", arr1.ndim)
    print("arr2.ndim:", arr2.ndim)
    print("arr3.ndim:", arr3.ndim)

    # shape 表示形状 返回一个元组  分别表示第一层括号有几个，第二层括号有几个
    print("arr1.shape:", arr1.shape)  # (6,)
    print("arr2.shape:", arr2.shape)  # (2,3)
    print("arr3.shape:", arr3.shape)  # (2,2,3)

    # 数组元素总个数
    print("arr1.size:", arr1.size)  # 6
    print("arr2.size:", arr2.size)  # 6
    print("arr3.size:", arr3.size)  # 12

    # 数组元素对象的类型，整数默认是int32，小数默认是double64
    print("arr1.dtype:", arr1.dtype)  # int32 表示每个元素用32bit存储，也就是4字节
    print("arr2.dtype:", arr2.dtype)  # int32
    print("arr3.dtype:", arr3.dtype)  # int32

    # 数组中每个元素所占字节大小
    print("arr1.itemsize:", arr1.itemsize)  # 4
    print("arr2.itemsize:", arr2.itemsize)  # 4
    print("arr3.itemsize:", arr3.itemsize)  # 4

    # astype方法改变dtype(data type)
    arr1 = arr1.astype("int16")
    arr2 = arr2.astype("float32")
    arr3 = arr3.astype("complex")

    print("arr1.dtype:", arr1.dtype)  # int16
    print("arr2.dtype:", arr2.dtype)  # float32
    print("arr3.dtype:", arr3.dtype)  # complex

    print("arr1:", arr1)
    print("arr2:", arr2)
    print("arr3:", arr3)


def base2():
    arr = np.arange(0, 16)

    print(type(arr))  # <class 'numpy.ndarray'>

    # print(arr) # [0 1 2 3 4]

    # print(arr.reshape(2,8))

    # print(arr.reshape(2,2,4))

    A = np.array([[1, 1], [0, 1]])

    B = np.array([[2, 1], [3, 4]])

    print(A + B)

    print(A - B)

    print(A * B)

    print(A / B)

    print(A > B)

    print(A == 0)

    print(A > 0)

    print(A < 0)

    # 矩阵乘法
    print(np.dot(A, B))


# 创建numpy.ndarray
def base3():
    arr1 = np.linspace(0, np.pi, 5)

    arr2 = np.arange(0, 5)

    # print(arr1)

    # print(arr2)

    arr3 = np.arange(0, 12).reshape(3, 4)

    print(arr3)

    # print(arr3.max())

    # print(arr3.max(axis=1))

    arr4 = np.zeros(10)
    arr5 = np.zeros((3, 3))
    arr6 = np.eye(5)
    arr7 = np.full((3, 3), 10)

    arr8 = np.random.rand(10, 10)
    # 返回一个随机数，不是数组 type是float
    arr9 = np.random.uniform(0, 100)

    print(arr4)
    print(arr5)
    print(arr6)
    print(arr7)
    print(arr8)
    print(arr9)
    print(type(arr9))


def base4():
    arr1 = np.arange(0, 20).reshape(4, 5)
    print(arr1)
    # 最大值
    print(np.amax(arr1, axis=1))  # [ 4  9 14 19]
    print(np.amax(arr1, axis=0))  # [15 16 17 18 19]

    # 最小值
    print(np.amin(arr1, axis=1))  # [ 0  5 10 15]
    print(np.amin(arr1, axis=0))  # [0 1 2 3 4]

    # 均值
    print(np.mean(arr1, axis=1))  # [ 2.  7. 12. 17.]
    print(np.mean(arr1, axis=0))  # [ 7.5  8.5  9.5 10.5 11.5]

    # 方差
    print(np.std(arr1, axis=1))  # [1.41421356 1.41421356 1.41421356 1.41421356]
    print(np.std(arr1, axis=0))  # [5.59016994 5.59016994 5.59016994 5.59016994 5.59016994]

    arr2 = np.arange(0, 10).reshape(2, 5)
    print(arr1)
    print(arr2)

    print(np.vstack((arr1, arr2)))  # 垂直拼接


base4()
