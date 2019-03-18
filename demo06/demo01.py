import tensorflow as tf

# NN的基本流程
def demo01():
    # 1. 图的定义阶段

    # 定义了两个常量op
    m1 = tf.constant([3,5])
    m2 = tf.constant([2,4])

    # 定义了加法op
    result = tf.add(m1,m2)

    # Tensor("Add:0", shape=(2,), dtype=int32)
    # print(result)

    # 2. 图的执行
    with tf.Session() as sess:
        res = sess.run(result)

    print(res)

# 常量
def demo02():
    # name 是tensor的唯一标识，冒号后边的是第几个输出结果，shape是维度，type是类型
    # 要保证参与运算的张量类型相一致
    a = tf.constant([[2.0,3.0]],name="a")
    b = tf.constant([[1.0],[4.0]],name="b")
    result = tf.matmul(a,b,name="mul")

    # Tensor("mul:0", shape=(1, 1), dtype=float32)
    print(result)

# 变量
def demo03():
    a = tf.Variable(3,name="a")
    b = tf.Variable(4,name="b")

    res = tf.add(a,b)
    # 有变量的话必须要进行初始化
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 必须先执行变量初始化操作
        sess.run(init)
        print(sess.run(res))

# 占位符和feed方式
def demo04():
    a = tf.placeholder(name="a",dtype=tf.float32)
    b = tf.placeholder(name="b",dtype=tf.float32)

    res = tf.matmul(a,b,name="res")

    # 等执行的时候 把占位符补全，通过feed_dict字典的方式
    with tf.Session() as sess:
        result = sess.run(res,feed_dict={a:[[2.,3.]],b:[[1.],[2.]]})
        print(result)

# fetch的用法，即我们利用session的run()方法同时取回多个tensor值
def demo05():
    a = tf.constant(5)
    b = tf.constant(6)
    c = tf.constant(4)

    add = tf.add(b,c)
    mul = tf.multiply(a,add)

    with tf.Session() as sess:
        result = sess.run([add,mul])
        print(result)

def demo06():
    import numpy as np
    # 真正模型
    x = np.random.rand(100)
    y = x * 0.1 + 0.2

    # 参数初始化
    k = tf.Variable(1.0,dtype=tf.float32)
    b = tf.Variable(2.0,dtype=tf.float32)
    y_ = k * x + b

    # 定义二次损失函数
    loss = tf.reduce_mean(tf.square(y - y_))

    # 定义梯度下降优化器 optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)

    # 最小化loss函数
    train = optimizer.minimize(loss,name="train")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(train)
        for i in range(201):
            if(i % 20 == 0):
                print(sess.run(k),sess.run(b),sess.run(loss))



def main():
    demo06()

if __name__ == "__main__":
    main()