# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import os
'''
防止报错：
I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 常量使用和Session


def demo01():
    # 创建两个常量op 一行两列 两行一列
    m1 = tf.constant([[3, 3]])
    m2 = tf.constant([[2], [3]])

    # 矩阵乘法的op
    product = tf.matmul(m1, m2)

    # Tensor("MatMul:0", shape=(1, 1), dtype=int32)
    # print(product)

    with tf.Session() as sess:
        result = sess.run(product)
        # [[15]]
        print(result)

# 变量要进行初始化


def demo02():
    x = tf.Variable([1, 2])
    a = tf.constant([3, 3])

    # 定义两个op
    sub = tf.subtract(x, a)
    add = tf.add(x, sub)

    # 进行变量初始化
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 执行变量初始化
        sess.run(init)
        print(sess.run(sub))
        print(sess.run(add))


def demo03():
    # 创建一个变量
    state = tf.Variable(0, name="count")

    # 创建一个op，作用是加1
    new_value = tf.add(state, 1)

    # tensorflow 赋值，将new_value赋值给state
    update = tf.assign(state, new_value)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(state))
        for i in range(5):
            sess.run(update)
            print(sess.run(state))

# fetch 就是一次性run多个op


def demo04():

    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)

    add = tf.add(input2, input3)
    mul = tf.multiply(add, input1)

    with tf.Session() as sess:
        result = sess.run([mul, add])
        print(result)

# feed 定义placeholder,运行时传值


def demo05():
    # 定义占位符,类型是tf.float32
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        # feed 的形式以字典的形式传入
        result = sess.run(output, feed_dict={input1: [7.0], input2: [2.0]})
        print(result)

# 一元线性回归


def demo06():
    # 造数据
    import numpy as np
    import matplotlib.pyplot as plt

    x_data = np.random.rand(100)
    y_data = x_data * 0.1 + 0.2

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data, color="red", marker="x")

    # 创建线性模型，并初始化参数
    b = tf.Variable(0.0)
    k = tf.Variable(0.0)
    y = x_data * k + b

    # 构造二次代价函数
    loss = tf.reduce_mean(tf.square(y - y_data))

    # 梯度下降优化loss的optimizer 优化器,学习率是0.2
    optimizer = tf.train.GradientDescentOptimizer(0.2)

    # 最小化代价函数
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(201):
            sess.run(train)
            if(i % 20 == 0):
                k_value, b_value = sess.run([k, b])
                #  输出k和b
                print("%d,k=%f,b=%f" % (i, k_value, b_value))
        # 得到预测值
        prediction_value = y.eval()

    # 画图比较一下
    plt.plot(x_data, prediction_value, color="blue")
    plt.legend()
    plt.show()

# 一元函数非线性回归


def demo07():
    import numpy as np
    import matplotlib.pyplot as plt
    # 造数据
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 定义神经网络中间层
    weight_L1 = tf.Variable(tf.random_normal([1, 10]))
    bias_L1 = tf.Variable(tf.zeros([1, 10]))
    output_L1 = tf.matmul(x, weight_L1) + bias_L1
    L1 = tf.nn.tanh(output_L1)

    # 定义神经网络输出层
    weight_L2 = tf.Variable(tf.random_normal([10, 1]))
    bias_L2 = tf.Variable(tf.zeros([1, 1]))
    output_L2 = tf.matmul(L1, weight_L2) + bias_L2
    prediction = tf.nn.tanh(output_L2)

    loss = tf.reduce_mean(tf.square(y - prediction))

    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            sess.run(train, feed_dict={x: x_data, y: y_data})

        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        plt.figure()
        plt.scatter(x_data, y_data, color="blue", marker="o")
        plt.plot(x_data, prediction_value, color="r")
        plt.show()

# 一元函数非线性回归
# 激活函数，优化器，学习率都会对模型产生影响，有时候模型很难回归


def demo08():

    import matplotlib.pyplot as plt
    import numpy as np
    x_data = np.linspace(-0.5, 0.5, 500)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.log(np.abs(x_data)) + np.exp(x_data) + \
        np.square(x_data) + noise

    # 输入是一个数，输出是一个数，忽略样本个数
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 定义神经网络中间层
    # x是1*1 w是1*100 相乘之后是1*100 加上偏置项1*100 因此中间层输出是1*100的
    weight_L1 = tf.Variable(tf.random_normal([1, 100]))
    bias_L1 = tf.Variable(tf.zeros([1, 100]))
    output_L1 = tf.matmul(x, weight_L1) + bias_L1
    L1 = tf.nn.tanh(output_L1)

    # 1*100的和100*10的相乘，得到的是1*10的
    weight_L2 = tf.Variable(tf.random_normal([100, 10]))
    bias_L2 = tf.Variable(tf.random_normal([1, 10]))
    output_L2 = tf.matmul(L1, weight_L2) + bias_L2
    L2 = tf.nn.tanh(output_L2)

    # 定义神经网络输出层
    # 上一层输出是1 * 100的，但是我想输出1*1的，因此w应该是100*1的，加上偏置项1*1的，输出就是1*1的
    weight_L3 = tf.Variable(tf.random_normal([10, 1]))
    bias_L3 = tf.Variable(tf.zeros([1, 1]))
    output_L3 = tf.matmul(L2, weight_L3) + bias_L3
    prediction = tf.nn.tanh(output_L3)

    # MSE
    loss = tf.reduce_mean(tf.square(y - prediction))

    learning_rate = 0.2
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if(i % 100 == 0):
                print("%d iterator,loss = %f" %
                      (i, sess.run(loss, feed_dict={x: x_data, y: y_data})))

        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        plt.figure()
        plt.scatter(x_data, y_data, color="blue", marker="o")
        plt.plot(x_data, prediction_value, color="r", lw=2)
        plt.show()

# 多元函数回归

def demo09():
    import matplotlib.pyplot as plt
    import numpy as np

    # 造数据
    x1_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    x2_data = np.linspace(-1.0, 1.0, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.01, x1_data.shape)
    y_data = np.square(x1_data) + np.square(x2_data) + noise

    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 1])

    # 输入是1*2  W是2*20
    weight_L1 = tf.Variable(tf.random_normal([2, 20]))
    bias_L1 = tf.Variable(tf.ones([1, 20]))
    output_L1 = tf.matmul(x, weight_L1) + bias_L1
    L1 = tf.nn.tanh(output_L1)

    # 输入是1*20  W是20*10
    weight_L2 = tf.Variable(tf.random_normal([20, 10]))
    bias_L2 = tf.Variable(tf.ones([1, 10]))
    output_L2 = tf.matmul(L1, weight_L2) + bias_L2
    L2 = tf.nn.tanh(output_L2)

    weight_L3 = tf.Variable(tf.random_normal([10, 1]))
    bias_L3 = tf.Variable(tf.ones([1, 1]))
    output_L3 = tf.matmul(L2, weight_L3) + bias_L3
    prediction = tf.nn.tanh(output_L3)

    loss = tf.reduce_mean(tf.square(y - prediction))

    learning_rate = 0.2
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(train, feed_dict={x: np.c_[x1_data, x2_data], y: y_data})
            if(i % 20 == 0):
                print("After %d Iteration" % i)
                print("loss = %f" % (
                    sess.run(loss, feed_dict={x: np.c_[x1_data, x2_data], y: y_data})))

        prediction_value = sess.run(prediction, feed_dict={
                                    x: np.c_[x1_data, x2_data]})

        plt.figure()
        plt.scatter(y_data,prediction_value,color="red",lw=2)

        refer_x = np.linspace(0.0,1.0,100)
        refer_y = refer_x

        plt.plot(refer_x,refer_y,color="blue")

        plt.xlabel("y")
        plt.ylabel("prediction")
        plt.show()


if __name__ == "__main__":
    demo09()
