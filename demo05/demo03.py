# matplotlib画图
# 散点图 折线图 条形图 直方图 饼状图 箱型图(即子图)
# 坐标轴调整 添加文字注释 patches

from matplotlib import pyplot as plt
import numpy as np


def wrapper(func):
    def inner():
        # 中文乱码的处理
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        func()

    return inner


# 散点图
def figure01():
    # 造数据
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([5, 2, 4, 2, 1, 4, 5, 2])

    # 画图 label 是图例
    # marker表示用什么符号标记那个点，s的大小表示marker的大小
    plt.scatter(x, y, label='point', color='r', s=10, marker="x")

    # 添加坐标轴label
    plt.xlabel('x')
    plt.ylabel('y')

    # 设置坐标轴范围
    plt.xlim([0, np.max(x) + 1])
    plt.ylim([0, np.max(y) + 1])

    # 设置title
    plt.title('a scatter figure')

    # 显示图例
    plt.legend()

    # 显示图片
    plt.show()


# 折线图
def figure02():
    # 造数据
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y1 = np.array([3, 5, 7, 6, 2, 6, 10, 15])
    y2 = np.array([4, 6, 8, 10, 23, 16, 3, 22])

    # 画图
    plt.plot(x, y1, color='r', label='y1')
    plt.plot(x, y2, color='g', label='y2')

    # 添加坐标轴label
    plt.xlabel('x')
    plt.ylabel('y')

    # 设置坐标轴范围
    plt.xlim([0, np.max(x) + 1])
    plt.ylim([0, np.max(y2) + 1])

    # 设置title
    plt.title('a line figure')

    # 显示图例
    plt.legend()

    # 显示图片
    plt.show()


# 饼图
def figure03():
    # 造数据
    slices = [7, 2, 2, 13]

    # 造图例和颜色
    activities = ['sleeping', 'eating', 'working', 'playing']
    cols = ['c', 'm', 'r', 'b']
    # explode 是个元组 用来表示拉出来那个切片,对应的数值表示拉出来的距离
    # autopct 表示显示百分比的格式

    # 画图
    plt.pie(slices,
            labels=activities,
            colors=cols,
            startangle=90,
            shadow=True,
            explode=(0, 0.1, 0.1, 0),
            autopct='%1.1f%%')

    # 设置title
    plt.title('a line figure')

    # 显示图例
    plt.legend()

    # 显示图片
    plt.show()


# 子图
def figure04():
    x = np.arange(0, 10)
    y1 = x
    y2 = 2 * x + 1

    plt.subplot(221)
    plt.plot(x, y1, label="y = x")
    plt.plot(x, y2, label="y = 2 * x + 1")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('a line figure')
    plt.legend()

    x = np.arange(-5, 6)
    y = np.abs(x)
    plt.subplot(222)
    plt.plot(x, y, label="y = abs(x)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-5, 5])
    plt.ylim([0, 5])
    plt.title('a line figure')
    plt.legend()

    plt.subplot(223)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([5, 2, 4, 2, 1, 4, 5, 2])
    plt.scatter(x, y, label='point', color='r', s=10, marker="x")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, np.max(x) + 1])
    plt.ylim([0, np.max(y) + 1])
    plt.title('a scatter figure')
    plt.legend()

    plt.subplot(224)
    slices = [7, 2, 2, 13]
    activities = ['sleeping', 'eating', 'working', 'playing']
    cols = ['c', 'm', 'r', 'b']
    plt.pie(slices,
            labels=activities,
            colors=cols,
            startangle=90,
            shadow=True,
            explode=(0, 0.1, 0.1, 0),
            autopct='%1.1f%%')
    plt.title('a line figure')
    plt.legend()

    plt.show()


# 高级条形图
def figure05():
    size = 5
    a = np.random.random(size)
    b = np.random.random(size)
    c = np.random.random(size)
    d = np.random.random(size)
    x = np.arange(size)

    total_width, n = 0.8, 3  # 有多少个类型，只需更改n即可
    width = total_width / n
    x = x - (total_width - width) / 2
    print(x)

    # bar(left, height, width=0.8,**wargs)
    plt.bar(x, a, width=width, label='a')
    plt.bar(x + width, b, width=width, label='b')
    plt.bar(x + 2 * width, c, width=width, label='c')

    plt.legend()
    plt.show()


# 直方图
def figure06():
    # 随机生成（10000,）服从正态分布的数据
    data = np.random.randn(10000)
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed/density:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(data, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7, label="Histogram")
    plt.xlabel("x")
    plt.ylabel("n or f")
    plt.title("Histogram")
    plt.legend()
    plt.show()


# 简单条形图
def figure07():
    fig, ax = plt.subplots()

    # 条形图的位置，即中点处
    position = np.arange(1, 6)

    # 造数据
    data = [9, 7, 5, 6, 8]

    # 画图
    ax.bar(position, data, 0.5, label="bar",color="#135446")

    # 显示图例
    plt.legend()

    # 显示图片
    plt.show()


figure07()
