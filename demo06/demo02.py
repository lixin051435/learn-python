import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import numpy as np
import warnings
# 不打印警告信息
warnings.filterwarnings("ignore")

# 变量初始化方式 会影响识别率的
def addConnect(inputs, in_size, out_size, activation_function=None):
    '''
    增加一层神经网络
    '''
    weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))

    bias = tf.Variable(tf.zeros([1, out_size]))

    y = tf.matmul(inputs, weight) + bias

    if activation_function is None:
        return y
    else:
        return activation_function(y)


def createImage(imageNumber):
    '''生成原始图片'''

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    save_dir = 'MNIST_data/raw/'

    if(os.path.exists(save_dir) is False):
        os.makedirs(save_dir)

    for i in range(imageNumber):
        image_array = mnist.train.images[i, :]

        image_array = image_array.reshape(28, 28)

        filename = save_dir + "mnist_train_%d.jpg" % (i)

        '''
        module 'scipy.misc' has no attribute 'toimage
        solution: pip3 install Pillow
        
        '''
        scipy.misc.toimage(image_array, cmin=0, cmax=1.0).save(filename)

        print("NO.%d image has been saved" % (i+1))


def createImageLabel(imageNumber):
    '''生成图像label'''

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    for i in range(imageNumber):

        one_hot_label = mnist.train.labels[i, :]

        label = np.argmax(one_hot_label)

        print("NO.%d image's label is %s" % (i+1, label))


def test_mnist():
    '''测试数据集target'''

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # train set (55000,784 = 28*28) (55000,10)
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)

    # validation set (5000, 784)
    print(mnist.validation.images.shape)
    print(mnist.validation.labels.shape)

    # test set (10000, 784)
    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)


def predictByDNN():
    '''
    用DNN识别mnist数据集
    别人写的 就一个输出层，没有激活函数
    '''

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 28*28])
    y = tf.placeholder(tf.float32, [None, 10])

    predict_y = tf.nn.softmax(addConnect(x, 28*28, 10))

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_y), 1))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(500):
        images, labels = mnist.train.next_batch(batch_size)
        session.run(train, feed_dict={x: images, y: labels})
        if i % 25 == 0:
            correct_prediction = tf.equal(
                tf.argmax(predict_y, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(
                accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))


def predictByDNN2():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 28*28])
    y = tf.placeholder(tf.float32, [None, 10])

    connect1 = addConnect(x, 28*28, 300, tf.nn.relu)
    connect2 = addConnect(connect1, 300, 64, tf.nn.relu)
    predict_y = addConnect(connect2, 64, 10, tf.nn.softmax)

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_y), 1))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(500):
        images, labels = mnist.train.next_batch(batch_size)
        session.run(train, feed_dict={x: images, y: labels})
        if i % 25 == 0:
            correct_prediction = tf.equal(
                tf.argmax(predict_y, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(
                accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))


def predict():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 100
    X_holder = tf.placeholder(tf.float32)
    y_holder = tf.placeholder(tf.float32)

    connect_1 = addConnect(X_holder, 784, 300, tf.nn.relu)
    predict_y = addConnect(connect_1, 300, 10, tf.nn.softmax)
    loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
    optimizer = tf.train.AdagradOptimizer(0.3)
    train = optimizer.minimize(loss)

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(5000):
        images, labels = mnist.train.next_batch(batch_size)
        session.run(train, feed_dict={X_holder: images, y_holder: labels})
        if i % 50 == 0:
            correct_prediction = tf.equal(
                tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(
                accuracy, feed_dict={X_holder: mnist.test.images, y_holder: mnist.test.labels})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))


if __name__ == "__main__":
    predictByDNN2()
