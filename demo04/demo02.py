from collections import Iterable


def fun1():
    from domain.user import User
    user = User("张三","zhangsan")
    print(user)

# datetime模块 时间相关
def test_datetime():
    import datetime
    i = datetime.datetime.now()
    print("当前的日期和时间是 %s" % i)
    print("ISO格式的日期和时间是 %s" % i.isoformat())
    print("当前的年份是 %s" % i.year)
    print("当前的月份是 %s" % i.month)
    print("当前的日期是  %s" % i.day)
    print("dd/mm/yyyy 格式是  %s/%s/%s" % (i.day, i.month, i.year))
    print("当前小时是 %s" % i.hour)
    print("当前分钟是 %s" % i.minute)
    print("当前秒是  %s" % i.second)
    import time
    time.sleep(2)
    print(datetime.datetime.now() - i)

def test_time():
    import time
    print(time.time())
    time.sleep(10)
    print(time.time())


# 一些集合类
# OrderedDict key有顺序的字典
# namedtuple 有名字的元组
# deque 比list更强大在列表
def test_collections():
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    print(p.x)
    print(p.y)

    from collections import deque
    q = deque(['a', 'b', 'c'])
    q.append('x')
    q.appendleft('y')
    print(isinstance(q, Iterable))
    q = iter(q)
    while True:
        try:
            print(next(q))
        except:
            break

if __name__ == '__main__':
    test_datetime()


