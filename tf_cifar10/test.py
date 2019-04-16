# 模拟tensorflow工作流程

# encoding:utf-8
import tensorflow as tf

import os

filenames = ['tf_cifar10/1.png', 'tf_cifar10/2.png', 'tf_cifar10/3.png']
# shuffle=False表示不打乱顺序，num_epochs=3表示整个队列获取三次
# queue 就是文件名队列
queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=3)

# 读取文件名队列中的数据
reader = tf.WholeFileReader()
key, value = reader.read(queue)
print("="*20)
print(key,value)
print("="*20)


with tf.Session() as sess:
    # 初始局部化变量,注意这个函数跟tf.global_variables_initializer.run()是不一样的
    # 因为string_input_producer函数的num_epochs=3传入的是局部变量
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        data = sess.run(value)
        # 如果文件夹不存在，则创建
        if not os.path.exists("shuffle_false"):
            os.makedirs("shuffle_false")

        with open('shuffle_false/image_%d.jpg' % i, 'wb') as fd:
            fd.write(data)

# 执行完之后会报OutOfRangeError异常,这就是epoch跑完，队列关闭的标志
