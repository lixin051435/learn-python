# python 基础和流程控制语句

print("hello world", "python and", "Anaconda")

# python 运算符
print("="*50)
print("3除以6的结果为：", 3/6)
print("3除以6的商为：", 3//6)
print("3除以6的余数为：", 3 % 6)

# 逻辑运算符 and or not
print(3 > 1 and 3 < 2)  # False
print(3 > 1 and 3 < 2)  # False
print(not 3 < 2)  # True

# python最牛逼的地方之一 多行字符串用三个单引号表示 r表示原生字符串，即没有什么转义字符之类的
str = r'''你好，我进行一下自我介绍
    我是李鑫，我正在学习python，
    python真好
    '''
print(str)

# 基本类型传递 是值传递
a = 2
b = a
a = 3

# <class int> 一切都是对象(id,type,value)
# 只是 int float 是不可变类型,不可变类型发生变化后会返回一个新对象
print(type(a), type(b))
# 1833289520 1833289488 id方法是取地址
print(id(a), id(b))
print("a = ", a, "b = ", b)  # a = 3 b = 2

a = 2.2
b = 3.3
b = a
a = 3.3

# <class float>
print(type(a), type(b))
# 4723312 4723360
print(id(a), id(b))
print("a = ", a, "b = ", b)

a = "123"
b = "321"
b = a
a = "12345"
# <class str>
print(type(a), type(b))
print(id(a), id(b))
# a = 12345 b = 123
print("a = ", a, "b = ", b)

a = 5
# 1833289584
print(id(a))
a += 1
# 1833289616
print(id(a))

# 分支结构
num = 5
if(num == 5):
    print('boss')
elif(num == 3):
    print('user')
elif(num == 1):
    print('worker')
elif(num < 0):
    print('error')
else:
    print('roadman')

# for循环结构
sum = 0
for i in range(1,101):
    sum += i
print(sum)

# while循环结构
i = 0
sum = 0
while(i < 101):
    sum += i
    i += 1
print(sum)


