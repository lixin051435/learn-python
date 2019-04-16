import cifar10

import tensorflow as tf

# tf.app.flags.FLAGS 是 TensorFlow内部的一个全局变量存储器，同时可以用于命令行参数的处理
FLAGS = tf.app.flags.FLAGS

# 在cifar10 模块中已经定义了f.app.flags.FLAGS.data_dir 为cifar10的数据路径
# 把这个路径修改为cifar10_data
FLAGS.data_dir = "cifar10_data/"

# 如果不存在数据文件，就会执行下载
cifar10.maybe_download_and_extract()