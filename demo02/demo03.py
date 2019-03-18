# python 异常处理
# logging 模块常常用做调试
import logging
logging.basicConfig(level=logging.INFO)

# python 所有异常的父类是BaseException


def test1(a, b):
    # logging.basicConfig(level=logging.INFO) 这句话不加不会输出
    # logging.info("a = %d,b = %d" % (a, b))
    try:
        c = a / b
    except BaseException as e:
        c = None
        print(e.__dict__)  # division by zero
        # logging.exception(e)  # 用logging记录错误信息
    finally:
        print("执行到finally")

    return c


# 定义一个异常


class MyException(BaseException):
    pass


# 抛出异常


def test2():
    raise MyException()


def main():
    print(test1(1, 2))

# import 这个模块就相当于执行了一遍这个模块 类似于导入JavaScript
import demo02

# __name__ 这个属性，如果这个python文件正在执行，那么这个脚本的__name__就是__main__,否则就是模块名
print(__name__)
print(demo02.__name__)

if __name__ == '__main__':
    main()
    try:
        test2()
    except BaseException as e:
        print(e)
