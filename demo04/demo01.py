from collections import Iterable


def test_Iterable():
    print(isinstance([], Iterable))  # True
    print(isinstance((), Iterable))  # True
    print(isinstance({}, Iterable))  # True
    print(isinstance((x ** 2 for x in range(100) if x % 3 == 0), Iterable))  # True
    print(isinstance('aaafds', Iterable))  # True
    print(isinstance(100, Iterable))  # False


def test_iter():
    a = [1, 2, 3, 4, 5, 6]
    print(isinstance(a, Iterable))
    a = iter(a)
    while True:
        try:
            print(next(a))
        except:
            print("迭代结束了")
            break


def wrapper(func):
    print("这里是wrapper")

    def inner():
        print("这里是inner")
        func()

    return inner


# @wrapper的过程就是 f = wrapper(f)
# 请读者自己考虑函数指针的指向问题
@wrapper
def f():
    print("这里是f")


def main():
    test_iter()


if __name__ == '__main__':
    main()
