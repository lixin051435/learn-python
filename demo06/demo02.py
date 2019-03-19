import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import numpy as np


def demo01():
    # 读取数据 并进行独热编码
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)
    # 学习率
    learning_rate = 0.2
    # 训练多少轮
    train_epochs = 20
    # 一次训练读取几张图片
    batch_size = 64
    # 每轮可以训练多少次
    total_batch = mnist.train.num_examples // batch_size

    # 输入层，隐藏层，输出层参数
    n_input = 28 * 28
    n_hidden1 = 512
    n_hidden2 = 256
    n_hidden3 = 128
    n_classes = 10

    # x是一行28*28列，y是一行10列
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    weight = {"L1": tf.Variable(tf.random_normal([n_input, n_hidden1])),
              "L2": tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
              "L3": tf.Variable(tf.random_normal([n_hidden2, n_hidden3])),
              "out": tf.Variable(tf.random_normal([n_hidden3, n_classes]))}

    bias = {"L1": tf.Variable(tf.random_normal([1, n_hidden1])),
            "L2": tf.Variable(tf.random_normal([1, n_hidden2])),
            "L3": tf.Variable(tf.random_normal([1, n_hidden3])),
            "out": tf.Variable(tf.random_normal([1, n_classes]))}

    L1 = tf.nn.sigmoid(tf.matmul(x, weight["L1"]) + bias["L1"])
    L2 = tf.nn.sigmoid(tf.matmul(L1, weight["L2"]) + bias["L2"])
    L3 = tf.nn.sigmoid(tf.matmul(L2, weight["L3"]) + bias["L3"])
    logits = (tf.matmul(L3, weight["out"]) + bias["out"])

    prediction = tf.nn.softmax(logits)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    pre_correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(pre_correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(train_epochs):
            for batch in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            if epoch % 10 == 0:
                loss_, acc = sess.run([loss, accuracy], feed_dict={
                                      x: batch_x, y: batch_y})
            print("epoch {},  loss {:.4f}, acc {:.3f}".format(epoch, loss_, acc))

        # 计算测试集的准确度
        test_acc = sess.run(accuracy, feed_dict={
                            x: mnist.test.images, y: mnist.test.labels})
        print('test accuracy', test_acc)

# 生成原始图片
def createImage(imageNumber):

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

# 生成图像label
def createImageLabel(imageNumber):

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    for i in range(imageNumber):

        one_hot_label = mnist.train.labels[i, :]

        label = np.argmax(one_hot_label)

        print("NO.%d image's label is %s" % (i+1, label))

# 测试数据集target
def test_mnist():

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

# 只有一个输出层的神经网络，通过softmax进行分类
def predictBySoftmax():

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # x表示待识别的图像
    x = tf.placeholder(tf.float32, [None, 784])
    # y_ 表示实际的图像label
    y_ = tf.placeholder(tf.float32, [None, 10])

     # 训练多少轮
    train_epochs = 20
    # 一次训练读取几张图片
    batch_size = 64
    # 每轮可以训练多少次
    total_batch = mnist.train.num_examples // batch_size

    # 输入层，隐藏层，输出层参数
    n_input = 28 * 28
    n_hidden1 = 512
    n_hidden2 = 256
    n_hidden3 = 128
    n_classes = 10

    weight = {"L1": tf.Variable(tf.random_normal([n_input, n_hidden1])),
              "L2": tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
              "L3": tf.Variable(tf.random_normal([n_hidden2, n_hidden3])),
              "out": tf.Variable(tf.random_normal([n_hidden3, n_classes]))}

    bias = {"L1": tf.Variable(tf.random_normal([1, n_hidden1])),
            "L2": tf.Variable(tf.random_normal([1, n_hidden2])),
            "L3": tf.Variable(tf.random_normal([1, n_hidden3])),
            "out": tf.Variable(tf.random_normal([1, n_classes]))}

    L1 = tf.nn.relu6(tf.matmul(x, weight["L1"]) + bias["L1"])
    L2 = tf.nn.relu6(tf.matmul(L1, weight["L2"]) + bias["L2"])
    L3 = tf.nn.relu6(tf.matmul(L2, weight["L3"]) + bias["L3"])
    logits = tf.matmul(L3, weight["out"]) + bias["out"]

    y = tf.nn.softmax(logits)

    # y 是模型输出的label，y_是真正的label y和y_误差越小越好 TensorFlow用交叉熵损失来描述
    # loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    loss = tf.reduce_mean(tf.square(y - y_))

    # 用梯度下降法优化损失 让损失减少
    learning_rate = 0.2
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss)

    # 创建回话

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(train_epochs):

            avg_loss = 0.0

            for batch in range(total_batch):
                # 从train中去100个训练数据
                # batch_xs 形状是（100,784） batch_ys (100,10)
                # 分别对应x和y_
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _,loss_value = sess.run([train_step,loss], feed_dict={x: batch_xs, y_: batch_ys})

                avg_loss = loss_value / total_batch

            if(epoch % 5 == 0):
                print("Iteration：%d，loss：%f"%(epoch,loss_value))


        # 正确预测结果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 计算预测准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 输出准确率 0.9148
        print("总共准确率为%f"%sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
    predictBySoftmax()
