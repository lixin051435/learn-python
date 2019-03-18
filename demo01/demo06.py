# python 内置函数第一波


# abs(Number) 取绝对值
a = 25.3
print("abs(%f)的值是%f" % (abs(a), a))

# divmod(a,b) (a//b,a % b)
a, b = 8, 5
print(divmod(a, b))

# int(数字，几进制=10) 转整数
# 如果带进制数的话，前面的数字要用字符串形式，并返回十进制的整数
# ord(字符) 返回对应十进制整数，即ASCII
print(int("F", 16))
print(ord("A"))

# str(object) 返回一个字符串对象
str = str([1, 2, 3, 4, "fds"])
# "[1, 2, 3, 4, 'fds']"
print(str, type(str))
print(str[0])

# pow(x,y) x的y次方 都是整数  相当于x**y
# math.pow(x,y) x,y是float
print(pow(2, 4))
import math

print(math.pow(2, 4))

# sum(可迭代对象)
print(sum([i ** 2 for i in range(1, 10)]))

# tuple(list) 将list转tuple
print(type(tuple([1, 2, 3])), tuple([1, 2, 3]))

# zip函数
names = ["张三", "李四", "王五"]
sexs = ["男", "男", "女"]
jobs = ["总裁", "经理", "老师"]
zipObj = zip(names, sexs, jobs)
for name, sex, job in zipObj:
    print(name, sex, job)

# reduce,map,filter函数

# reduce函数必须引入functools,两两迭代
from functools import reduce

sum = reduce(lambda a, b: a * b, [1, 2, 3, 4, 5])
print(sum)

# filter函数第一个参数函数必须返回值为bool 返回一个filter对象，可以转成list
iter = filter(lambda a: a % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(list(iter))

# map函数遍历序列，对序列中每个元素进行操作，最终返回map对象，可转成list，类似于列表生成式
print(list(map(lambda x: x ** 2, [1, 2, 3, 4, 5])))

# dir(对象) 返回当前变量的方法，变量等
print(dir(1))

# input(提示信息) 返回输入的字符串
# type(变量名) 返回类型
inputString = input("请输入一段字符串\n")
print(inputString, type(inputString))
