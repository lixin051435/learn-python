
# for 迭代字符串
for i in "abcd":
    print(i)
print("="*30)

# for 迭代list
for i in [1,"23",{"a":12}]:
    print(i)
print("="*30)

# for 迭代tuple
for i in (1,2,3):
    print(i)
print("="*30)

# for 迭代dict
for k,v in {"1":20,"aaa":"1978"}.items():
    print(k,v)
print("="*30)

# for 迭代 range对象 左闭右开
for i in range(0,100,2):
    print(i)
print("="*30)

# for 用zip并行迭代 zip函数会将多个可迭代对象转成tuple
names = ["张三","李四","王五"]
sexs = ["男","男","女"]
jobs = ["总裁","经理","老师"]
# zip(names,sexs,jobs)  就是(("张三","男","总裁"),("李四","男","经理"),("王五","女","老师"))
for name,sex,job in zip(names,sexs,jobs):
    print(name,sex,job)
print("="*30)

# 列表生成式 语法： [表达式 for item in 可迭代对象 if 判断条件]
print([x for x in range(1,100) if x % 3 == 0])
print([x**2 for x in range(1,100) if x % 3 == 0])
print([(row,col) for row in range(1,10) for col in range(1,10)])
print("="*30)

# 字典生成式 语法：{key:value for item in 可迭代对象 if 判断条件}
text = "faiuow34sd72hfsd"
print({c:text.count(c) for c in text})
print("="*30)

# 集合生成式 语法： {表达式 for item in 可迭代对象 if 判断条件}
print({x**2 for x in range(1,100) if x % 3 == 0})
print("="*30)

# 元组没有生成式 只返回迭代对象 迭代对象只能用一次 第二次开始就没有数据了
a  = (x for x in range(1,100) if x % 3 == 0)
print(a)
for i in a:
    print(i)
print("="*30)
for i in a:
    print(i)
