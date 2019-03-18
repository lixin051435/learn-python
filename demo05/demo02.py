# numpy 的切片操作
import numpy as np


def slice1():
    # 类似于list的range()
    arr = np.arange(10)
    print("arr的类型是%s" % (type(arr)))
    print(arr)
    print("arr[4]的值是%s" % (arr[4]))

    # 切片
    print("arr[3:6]的值是%s" % (arr[3:6]))
    arr_old = arr.copy()  # 先复制一个副本
    print(arr_old)
    arr[3:6] = 33

    # 可以发现将标量赋值给一个切片时，该值可以传播到整个选区
    print(arr)
    print(arr_old)


def slice2():
    arr = np.arange(12).reshape(3, 4)
    print("arr=", arr)

    # 行索引1-2  列索引1-3
    # 左闭右开
    print("arr[1:2,1:3]=", arr[1:2, 1:3])

    # 取第一维的全部
    # 按步长为2取第二维的索引0到末尾之间的元素
    print("arr[:, ::2]=", arr[:, ::2])


def slice3():
    # 生成6*6的数组
    arr3 = np.arange(36).reshape(6, 6)

    print(arr3)

    print(arr3[arr3 > 10])

    arr3[arr3 > 10] = 100

    print(arr3)


def slice4():
    # 模仿RGB图像的ndarray

    arr = np.arange(120).reshape(10, 4, 3)

    print(arr)

    # 第0个二维数组（也就是RGB图像的第一行）
    # 这个二维数组中的第0-1行 也就是第一行的前两个一维数组
    # 从这个一维数组中取得下标为2到4的数
    print(arr[0, 0:2, 2:5])


slice4()
