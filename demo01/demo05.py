
def test():
    '''我是test函数'''
    pass


# 函数注释写里面
print(test.__doc__)

def fun1():
    return 5

a = fun1
print(a())

def fun2(a):
    # 1833290384
    print(id(a))
    a = 20
    # 1833290064
    print(id(a))


a = 30
print(a)
# 1833290384
print(id(a))

fun2(a)
print(a)
# 1833290384
print(id(a))

import copy
def testCopy():
    a = [10,20,[5,6]]
    b = copy.copy(a)
    print(a)
    print(b)
    b.append(30)
    b[2].append(7)
    print("浅拷贝")
    print(a)
    print(b)

def testDeepCopy():
    a = [10,20,[5,6]]
    b = copy.deepcopy(a)
    print(a)
    print(b)
    b.append(30)
    b[2].append(7)
    print("深拷贝")
    print(a)
    print(b)

testCopy()
testDeepCopy()


func = lambda a,b,c:a + b + c
print(func(1,2,3))