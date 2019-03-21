# tensorflow 加载模型
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import warnings
import math
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def line_regression():
    """
    desc:
        y = 0.1x + 0.2 一元线性模型回归，并把模型保存到models文件夹
    """

    x_data = np.random.rand(100)
    y_data = x_data * 0.1 + 0.2

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data, color="red", marker="x")

    # 创建线性模型，并初始化参数
    b = tf.Variable(0.0, name="b")
    k = tf.Variable(0.0, name="k")
    y = x_data * k + b

    # 构造二次代价函数
    loss = tf.reduce_mean(tf.square(y - y_data))

    # 梯度下降优化loss的optimizer 优化器,学习率是0.2
    optimizer = tf.train.GradientDescentOptimizer(0.2)

    # 最小化代价函数
    train = optimizer.minimize(loss)

    # 只保存最后一个
    saver = tf.train.Saver(max_to_keep=1)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(201):
            sess.run(train)
            if(i % 20 == 0):
                k_value, b_value = sess.run([k, b])
                #  输出k和b
                print("%d,k=%f,b=%f" % (i, k_value, b_value))
                # global_step 参数是迭代次数，会放到文件名后边
                saver.save(sess, "models/line_regression")
        # 得到预测值
        prediction_value = y.eval()

    # 画图比较一下
    plt.plot(x_data, prediction_value, color="blue")
    plt.legend()
    plt.show()


def load_line_regression_model():
    """
    desc:
       加载line_regression训练好的模型并获取tensor
    """

    with tf.Session() as sess:
        # 加载模型 import_meta_graph填的名字meta文件的名字
        saver = tf.train.import_meta_graph("models/line_regression.meta")

        # 检查checkpoint，所以只填到checkpoint所在的路径下即可，不需要填checkpoint
        saver.restore(sess, tf.train.latest_checkpoint("models"))

        # 根据name获取变量值
        k = sess.run("k:0")
        b = sess.run("b:0")

        print("k = %f,b = %f" % (k, b))


def non_line_regression():
    """
    desc:
        非线性回归: y = x * x 
    """
    # 造数据
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1], name="x")
    y = tf.placeholder(tf.float32, [None, 1], name="y")

    # 定义神经网络中间层
    weight_L1 = tf.Variable(tf.random_normal([1, 10]))
    bias_L1 = tf.Variable(tf.zeros([1, 10]))
    output_L1 = tf.matmul(x, weight_L1) + bias_L1
    L1 = tf.nn.tanh(output_L1)

    # 定义神经网络输出层
    weight_L2 = tf.Variable(tf.random_normal([10, 1]))
    bias_L2 = tf.Variable(tf.zeros([1, 1]))
    output_L2 = tf.matmul(L1, weight_L2) + bias_L2
    prediction = tf.nn.tanh(output_L2, name="predict")

    loss = tf.reduce_mean(tf.square(y - prediction))

    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if(i % 100 == 0):
                loss_value = sess.run(loss, feed_dict={x: x_data, y: y_data})

                print("step:%d,loss:%f" % (i, loss_value))
                saver.save(sess, "models/non_line_regression")

        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        plt.figure()
        plt.scatter(x_data, y_data, color="blue", marker="o")
        plt.plot(x_data, prediction_value, color="r")
        plt.show()


def load_non_line_regression():
    """
    desc:
        加载non_line_regression产生的模型并预测
    """
    # 造数据
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    y_data = np.square(x_data)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/non_line_regression.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models"))

        graph = tf.get_default_graph()

        # 得到的就是那两个placeholder
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        
        # 得到的就是最后的输出
        predict = graph.get_tensor_by_name("predict:0")

        predict_value = sess.run(predict, feed_dict={x: x_data})

        plt.figure()
        plt.scatter(x_data, y_data, color="blue", marker="o", label="actual")
        plt.scatter(x_data, predict_value, color="r", label="pred")
        plt.legend()
        plt.show()


def train_model(epochs=5000, learning_rate=0.1):
    """
    desc:
        用DNN进行回归，并保存模型
    """

    x_data = np.linspace(-1.0, 1.0, 500)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    # y = x * x 这个模型
    y_data = np.square(x_data) + noise

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1], name="x")
    y = tf.placeholder(tf.float32, [None, 1], name="y")

    input_num = 1
    hidden1_num = 50
    hidden2_num = 10
    output_num = 1

    weights = {"L1": tf.Variable((tf.random_normal([input_num, hidden1_num]))),
               "L2": tf.Variable((tf.random_normal([hidden1_num, hidden2_num]))),
               "out": tf.Variable((tf.random_normal([hidden2_num, output_num])))}

    bias = {"L1": tf.Variable((tf.zeros([1, hidden1_num]))),
            "L2": tf.Variable((tf.zeros([1, hidden2_num]))),
            "out": tf.Variable((tf.zeros([1, output_num])))}

    L1 = tf.nn.tanh(tf.matmul(x, weights["L1"]) + bias["L1"])
    L2 = tf.nn.tanh(tf.matmul(L1, weights["L2"]) + bias["L2"])
    predict = tf.nn.tanh(
        tf.matmul(L2, weights["out"]) + bias["out"], name="predict")

    error = y - predict
    loss = tf.reduce_mean(tf.square(error))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # max_to_keep=1 保存最后1个模型 会根据训练次数 一个一个覆盖的
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if(epoch % 10 == 0):
                print("step:%d,loss:%f" % (epoch, sess.run(
                    loss, feed_dict={x: x_data, y: y_data})))
                saver.save(sess, "models/y=x2", global_step=epoch)

        predict_value = sess.run(predict, feed_dict={x: x_data})
        plt.figure()
        plt.scatter(x_data, y_data, marker="x", color="red")
        plt.plot(x_data, predict_value, color="blue")
        plt.legend()
        plt.show()

        value = sess.run(predict, feed_dict={x: [[0.3]]})
        print("0.3的预测值为%f" % value)


def load_model():
    """
    desc:
        恢复train_model函数的文件，并进行测试
        .meta文件：一个协议缓冲，保存tensorflow中完整的graph、variables、operation、collection
        checkpoint文件：一个二进制文件，包含了weights, biases, gradients和其他variables的值。
        但是0.11版本后的都修改了，用.data和.index保存值，用checkpoint记录最新的记录。
    """

    x_data = np.array([[0.2], [0.3], [-0.3], [0.15], [-0.21], [0.22]])
    y_data = np.square(x_data)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/y=x2-4990.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models"))

        graph = tf.get_default_graph()
        # 得到两个placeholder 和 预测值
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        predict = graph.get_tensor_by_name("predict:0")

        # 进行预测
        predict_value = sess.run(predict, feed_dict={x: x_data})

        # 看看每个数据的差距多大
        print(np.abs(predict_value - y_data))

        plt.figure()
        plt.scatter(x_data, y_data, marker="x", color="red", label="actual")
        plt.scatter(x_data, predict_value, color="blue", label="pred")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    load_model()
