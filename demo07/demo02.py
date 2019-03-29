import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist_model(batch_size):
    # 读取模型 测试mnist图片
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    images, labels = mnist.test.next_batch(batch_size)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            "models/mnist_conv/mnist_conv.ckpt.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/mnist_conv"))

        print("="*10)

        graph = tf.get_default_graph()
        # 得到两个placeholder 和 预测值
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        dropout = graph.get_tensor_by_name("dropout:0")
        predict = graph.get_tensor_by_name("predict:0")

        predict_values = sess.run(predict, feed_dict={x: images, dropout: 1})

        for i in range(batch_size):
            predict_value = np.argmax(predict_values[i])
            label = np.argmax(labels[i])
            print("第%d张图片,预测值:%d,真实值:%d" % (i+1, predict_value, label))




def test_my_images(batch_size=10):
    # 读取模型 测试自己的图片
    images, labels = create28x28Image(batch_size)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            "models/mnist_conv/mnist_conv.ckpt.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/mnist_conv"))

        print("="*10)

        graph = tf.get_default_graph()
        # 得到两个placeholder 和 预测值
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        dropout = graph.get_tensor_by_name("dropout:0")
        predict = graph.get_tensor_by_name("predict:0")

        predict_values = sess.run(predict, feed_dict={x: images, dropout: 1})

        for i in range(batch_size):
            predict_value = np.argmax(predict_values[i])
            label = np.argmax(labels[i])
            print("第%d张图片,预测值:%d,真实值:%d" % (i+1, predict_value, label))


class MnistImage:
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __str__(self):
        return "image shape is %s,label is %s" % (self.image.shape, self.label)

    def toStandardImage(self):
        return self.image / 255

    def toStandardLabel(self):
        temp_label = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        temp_label[int(self.label)] = 1.0
        return temp_label


def create28x28Image(batchsize=10):
    import cv2
    import operator
    # 自己画的图片
    root = "MNIST_data/my_mnist_test/"

    mnistImageList = []
    if(os.path.exists(root)):
        dirList = os.listdir(root)
    for dirName in dirList:
        for i in range(5):
            # 拼接图片路径
            imagePath = root + "%s/%d.png" % (dirName, i)
            # 读取原始图像,用灰度图表示
            image = cv2.imread(imagePath, 0)
            # 反色变换，因为mnist图像就是黑底白字 并归一化，mnist图像像素点是[0,1]
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    image[row][column] = 255-image[row][column]

            if(operator.eq(image.shape, (28, 28))):
                 # 封装成对象
                mnistImage = MnistImage(image, float(dirName))
                mnistImageList.append(mnistImage)
                cv2.imwrite(imagePath, image)
            else:
                # 缩放到28*28像素
                new_image = cv2.resize(image, (28, 28), cv2.INTER_LINEAR)
                # 封装成对象
                mnistImage = MnistImage(new_image, float(dirName))
                mnistImageList.append(mnistImage)
                # 回写到原来图像路径中
                cv2.imwrite(imagePath, new_image)

    images = []
    labels = []
    for mnistImage in mnistImageList:
        # 转成784向量
        images.append(mnistImage.toStandardImage().reshape([784]))
        labels.append(mnistImage.toStandardLabel())

    return images[0:batchsize], labels[0:batchsize]


if __name__ == "__main__":
    test_my_images(50)
    # create28x28Image()
    # load_mnist_model(2)
