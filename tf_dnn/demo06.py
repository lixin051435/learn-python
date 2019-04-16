import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

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


learning_rate = 0.5
batch_size = 100
epochs = 500
step = 50

x = tf.placeholder(tf.float32, [None, 28*28], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")
dropout = tf.placeholder(tf.float32, name="dropout")

input_num = 28*28
hidden1_num = 300
hidden2_num = 64
output_num = 10

L1 = addConnect(x, input_num, hidden1_num, tf.nn.relu)
L2 = addConnect(L1, hidden1_num, hidden2_num, tf.nn.relu)
L3 = addConnect(L2, hidden2_num, output_num)
predict = tf.nn.softmax(L3, name="predict")

# loss = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict), 1))

# loss = tf.reduce_mean(tf.square(y - predict))

# loss = tf.reduce_mean(
#     tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        images, labels = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: images, y: labels})
        if(i % step == 0):
            correct_prediction = tf.equal(
                tf.argmax(predict, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            accuracy_value = sess.run(
                accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, dropout: 0.8})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))
