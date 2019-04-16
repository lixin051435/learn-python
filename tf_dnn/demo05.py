# 训练mnist并保存模型，利用模型进行预测

# 深入mnist数据集

import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import warnings
import math
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def addConnect(input, input_size, output_size, activate_function=None):
    """
    Args：
        input：输入的数据
        input_size：输入的数据维度，也就是说一行几列
        output_size: 输出维度，经过这层网络要输出一行几列的数据
        activate_function： 激活函数
    desc：
        构建一层NN
    """
    weight = tf.Variable(tf.truncated_normal(
        [input_size, output_size], stddev=0.1))
    bias = tf.Variable(tf.zeros([1, output_size]))
    y = tf.add(tf.matmul(input, weight), bias)
    if activate_function is None:
        return y
    else:
        return activate_function(y)


def drawDigit(position, image, title, isTrue=True):
    """
    desc:
        封装plt对image的画图

    args：
        position：展示在plt.subplot的哪个位置
        image：1 * 784 的ndarray
        title：plot的title,
        isTrue: 预测的是否是真
    """
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap="gray_r")
    plt.axis("off")
    if not isTrue:
        plt.title(title, color="red")
    else:
        plt.title(title)


def batchDraw(batch_size):
    """
    desc:
        批量图展示到plt上
    args：
        batch_size: 一次性展示多少张图，要完全平方数
    """
    # mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # 从train获取batch_size 这么多个数据
    images, labels = mnist.train.next_batch(batch_size)

    # 有多少个图片，其实就是batch_size
    image_number = images.shape[0]

    # 想吧batch_size个图片放到一个大正方形展示，那么有sqrt(batch_size)行和列
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number

    # 指定画布尺寸
    plt.figure(figsize=(row_number, column_number))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index+1)
                image = images[index]
                title = '%d' % (np.argmax(labels[index]))
                drawDigit(position, image, title)


def predictByDNN(epochs=1000, batch_size=100):
    """
    desc:
        用NN和softmax识别mnist数据集
    args：
        epochs：训练次数
        batch_size: 一批次训练多少张图片
    """

    x = tf.placeholder(tf.float32, [None, 28*28], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")

    connect1 = addConnect(x, 28*28, 300, tf.nn.relu)
    connect2 = addConnect(connect1, 300, 64, tf.nn.relu)
    predict_y = addConnect(connect2, 64, 10)
    predict_y = tf.nn.softmax(predict_y, name="predict")

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_y), 1))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    # 初始化变量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            images, labels = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: images, y: labels})
            if(epoch % 50 == 0):
                correct_prediction = tf.equal(
                    tf.argmax(predict_y, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
                accuracy_value = sess.run(
                    accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

                print('step:%d accuracy:%.4f' % (epoch, accuracy_value))
                saver.save(sess, "models/mnist/mnist_model")

        # 用测试集 可视化一下
        images, labels = mnist.test.next_batch(batch_size)
        predict_labels = sess.run(predict_y, feed_dict={x: images, y: labels})
        image_number = images.shape[0]
        row_number = math.ceil(image_number ** 0.5)
        column_number = row_number
        plt.figure(figsize=(row_number + 10, column_number + 10))
        for i in range(row_number):
            for j in range(column_number):
                index = i * column_number + j
                if index < image_number:
                    position = (row_number, column_number, index+1)
                    image = images[index]
                    actual = np.argmax(labels[index])
                    predict = np.argmax(predict_labels[index])
                    isTrue = actual == predict
                    title = 'act:%d--pred:%d' % (actual, predict)
                    drawDigit(position, image, title, isTrue)


def load_mnist_model(batch_size):
    images, labels = mnist.test.next_batch(batch_size)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/mnist/mnist_model.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/mnist"))

        graph = tf.get_default_graph()
        # 得到两个placeholder 和 预测值
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        predict = graph.get_tensor_by_name("predict:0")

        predict_values = sess.run(predict, feed_dict={x: images})

        for i in range(batch_size):
            predict_value = np.argmax(predict_values[i])
            label = np.argmax(labels[i])
            print("第%d张图片,预测值:%d,真实值:%d" % (i+1, predict_value, label))


if __name__ == "__main__":
    # predictByDNN()
    # plt.show()
    load_mnist_model(20)
