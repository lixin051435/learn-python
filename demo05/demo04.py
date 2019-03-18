import pandas as pd
from pandas import Series


# Series 数据结构 基本属性和ndarray一样
def test01():
    # 用list生成Series
    series = Series([100, 200])
    print(series)

    # 用Map生成Series
    series = Series({'2010': 100, '2012': 200, '2014': 300})
    print(series)

    print("数据索引", series.index)
    print(type(series.index))  # <class 'pandas.core.indexes.base.Index'>

    print("数据值", series.values)
    print(type(series.values))  # <class 'numpy.ndarray'>

    print('数据大小：', series.size)

    print('数据类型：', series.dtype)

    print('数据维度：', series.ndim)

    print('数据形状：', series.shape)

    # series 遍历
    for i in series:
        print(i)

    print(series['2010'])
    # 会保留numpy的运算
    print(2 * series)

    series2 = Series({'2010': 100, '2012': 200, '2013': 300})

    # 相同索引值进行运算，如有有一方没有的话，就是NaN，NaN需要用pd.isnull 或 pd.isna 判断，不能用is或==判断
    series3 = series2 + series
    print(pd.isna(series3['2014']))
    print(pd.isnull(series3['2014']))


# DataFrame 数据结构，可以认为是Series容器
def test02():
    from pandas import Series, DataFrame

    # data是字典类型
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

    frame  = DataFrame(data)

    '''
       pop   state  year
    0  1.5    Ohio  2000
    1  1.7    Ohio  2001
    2  3.6    Ohio  2002
    3  2.4  Nevada  2001
    4  2.9  Nevada  2002
    
    这里的0 1 2 3 是自动生成的索引
    '''
    # print(frame)

    # <class 'pandas.core.frame.DataFrame'>
    # print(type(frame))

    # columns 可以改变顺序，index可以制定索引
    frame2 = DataFrame(data=data,columns=["year","state","pop"],index=[2,3,4,5,6])

    # 获取某一列
    frame3 = frame["state"]

    # 获取某一行
    print(frame.ix[0])

    # 获取某几行
    print(frame.ix[range(3)])

    # 获取交叉值 先指定行再指定列
    print(frame.ix[0]['pop'],type(frame.ix[0]))

    # 获取交叉值 先指定列再指定行
    print(frame['pop'],type(frame['pop']))

    # 增加一列，每一行都是10
    frame["another"] = 10
    print(frame)

    frame["another"] = range(5)
    print(frame)

    # 判断another列是否为1
    '''
    0    False
    1     True
    2    False
    3    False
    4    False
    '''
    # 把frame.another大于1的值替换成200
    frame[frame.another > 1] = 200
    print(frame)

# 用pandas 读取csv或者excel
def test03():
    import numpy as np
    import pandas as pd

    # 返回值直接是DataFrame
    dataset = pd.read_csv("./data/Data.csv")
    print("数据基本信息：",dataset.info())
    print("数据表的维度：",dataset.shape)
    # values 转换为二维数组ndarray
    # print(dataset.values)

    # loc函数 通过行索引index中具体指来获取数据
    print(dataset.loc[1])

    # iloc函数 通过行号来获取数据 返回值是DataFrame
    x_values = dataset.iloc[:,:-1]
    y_values = dataset.iloc[:,-1]

    # 转换成ndarray
    x = x_values.values
    y = y_values.values

if __name__ == '__main__':
    test03()
