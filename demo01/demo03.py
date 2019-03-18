# 数据结构

def list_test():
    list = []
    # list 增 append(obj)  insert(index,obj)
    list.append(1)
    list.append(2)
    list.insert(1,"hello")
    list.append(4)
    list.append(5)
    print(list)

    # 切片
    print("list[:5]=", list[:5])  # 从头开始到下标为5结束，如果超过length，就到最后
    print("list[1:3]=", list[1:3])  # 从下标为1 到下标为3 左闭右开区间
    print("list[:-1]=", list[:-1])  # -1表示从逆序第一个
    print("list[4:-1]=", list[4:-1])  # []
    print("list[:]=", list[:])  # 全部

    # len函数 len(list)
    print("len(list) = ", len(list))

    # 列表拼接
    a = [1, 2, 3, 4]
    b = [1, 2, 3]
    print("a + b =", a + b)

    # in 关键字
    print(1 in list)  # True

    # list 删 remove(obj) pop(index)
    print(list)
    list.remove(2)
    print(list)

    # 扩充列表 extend(list)
    print(list)
    list.extend([1, 2, 3])
    print(list)

    # 查找 index(obj)
    print(list.index(4))  # 2
    # print(list.index("hwloo")) # 报错 因为hwloo不在list中，并不会返回-1

    # 引用传递
    lista = [1, 2, 3]
    listb = lista
    listb.extend(["fds", "fds", 434])
    print(lista)
    print(listb)

# python 元组 tuple，不可变类型
def tuple_test():
    tup1 = ('Google', 'Runoob', 1997, 2000)
    tup2 = (1, 2, 3, 4, 5 )
    tup3 = "a", "b", "c", "d"   #  不需要括号也可以
    print(type(tup1),type(tup2),type(tup3))

    # 元组里面只有一个元素 需要加逗号
    a = (50)
    b = (50,)
    c = ()
    print(type(a),type(b),type(c))  # int tuple tuple

    # 元组一经声明就不能修改 删除的话只能把整个元组删除
    del b


# python 字典 也就是Map
def dict_test():
    person = {
        "name" : "小名",
        "id" : 10010,
        "sex" : "男",
        "school" : "清华大学"
    }

    # 判断key是否在dict中 in
    print("aaa" in person)

    # obj[key] 如果key不存在 这样会异常
    # print(person["name"])

    # obj.get(key) 如果key不存在 返回None
    print(person.get("name"))
    print(person.get("namea")) # None

    # 删除key dict.pop(key)
    person.pop("name")
    print(person)

    # 声明一个空dict
    _dict = {}
    print(type(_dict))



# python 集合 set
def set_test():
    # 用set方法 把list，tuple转set
    a = set([1,2,3,4,5,2,3])
    print(a,type(a))
    # 创建一个空set
    b = set()
    print(b,type(b))

    # set.add(key)
    b.add(12)
    b.add(1)
    b.add(12)
    print(b)

    # set.remove(key)
    b.remove(12)
    print(b)

    # 集合的运算 同数学中集合运算
    a = [1,2,2,'a','a','d','e']
    b = [1,2,2,'a','a','b','b']
    c = set(a)
    d = set(b)
    print(c,d)
    # The result is {1, 2, 'a', 'd', 'e'} {'b', 1, 2, 'a'}
    e = c.intersection(d) # "取交集" “equals the command: c & b”
    f = c.union(d) #"并集" “equals the command: c \ d”
    g = c.difference(d) #"差集" “equald the command c - d”
    print(e,f,g)
    # The result is {'a', 1, 2} {1, 2, 'a', 'b', 'd', 'e'} {'d', 'e'}
    h = c.symmetric_difference(d) #"对称差集" “equals to c ^ d”
    i = c.issubset(d) # "判读是否为子集"
    j = c.issuperset(d) # "判读是否为超集"
    k = c.isdisjoint(d)#检查是否有相同元素,没有返回True
    print(h,i,j,k)
