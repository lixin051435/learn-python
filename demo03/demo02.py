import multiprocessing
import time
import os
import threading


def work(i):
    print("我的pid是：%s,ppid是%s" % (os.getpid(), os.getppid()))
    print("我被执行了", "我的参数是：", str(i))


# Process 基本测试
def Process_test1():
    p1 = multiprocessing.Process(target=work, args=(1,))
    p2 = multiprocessing.Process(target=work, args=(2,))

    p1.start()
    p2.start()

    print("p1.name:", p1.name)
    print("p2.name:", p2.name)

    print("p1.pid:", p1.pid)
    print("p2.pid:", p2.pid)

    p1.join()  # 父进程等待子进程结束
    p2.join()  # 父进程等待子进程结束
    print("子进程已经结束了")


class MProcess(multiprocessing.Process):
    def __init__(self, i):
        multiprocessing.Process.__init__(self)
        self.i = i

    # 重写run方法
    def run(self):
        print("子进程(%s) 开始执行，父进程为（%s）" % (os.getpid(), os.getppid()))
        t_start = time.time()
        # print("开始时间：",str(t_start))
        print("我被执行了,我的参数是：", str(self.i))
        t_stop = time.time()
        # print("结束时间：",str(t_stop))
        print("(%s)执行结束，耗时%0.2f秒" % (os.getpid(), t_stop - t_start))


def Process_test2():
    p1 = MProcess(1)
    p2 = MProcess(2)
    p3 = MProcess(3)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    print("子进程全部结束")


# 进程池
def pool_test():
    pool = multiprocessing.Pool(3)

    for i in range(0, 1000):
        # 向进程池中添加任务
        # 注意：如果添加的任务数量超过了　进程池中进程的个数的话，那么不会导致添加不进入
        # 添加到进程中的任务　如果还没有被执行的话，那么此时　他们会等待进程池中的
        # 进程完成一个任务之后，会自动的去用刚刚的那个进程　完成当前的新任务
        pool.apply_async(work, (i,))

    pool.close()  # 关闭进程池，相当于　不能够再次添加新任务了
    pool.join()  # 主进程　创建／添加　任务后，主进程　默认不会等待进程池中的任务执行完后才结束
    # 而是　当主进程的任务做完之后　立马结束，，，如果这个地方没join,会导致
    # 进程池中的任务不会执行
    print("主进程结束")


class MThread(threading.Thread):
    def __init__(self, i):
        threading.Thread.__init__(self)
        self.i = i

    def run(self):
        t_start = time.time()
        print("我被执行了,我的参数是：", str(self.i))
        t_stop = time.time()
        print("当前进程：(%s)执行结束，耗时%0.2f秒" % (os.getpid(), t_stop - t_start))


# 多线程基本测试
def thread_test1():
    t1 = MThread(1)
    t2 = MThread(2)
    t3 = MThread(3)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("子线程全部结束")


def main():
    pool_test()


if __name__ == '__main__':
    main()