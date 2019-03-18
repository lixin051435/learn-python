# 面向对象基础
class Person:
    def __init__(self, id):
        self.id = id


class Student(Person):
    number = 0

    # __inti__方法 是初始化方法，是Java里构造方法的一部分
    # __id,__name 是私有属性，其实也能拿得到
    def __init__(self, id, name, score):
        Person.__init__(self, id)
        self.__id = id
        self.__name = name
        self.score = score
        Student.number += 1

    @classmethod
    def f(cls):
        print("我是classmethod")

    @staticmethod
    def add(a, b):
        print("我是staticmethod")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    # 实例方法 必须用过对象调用
    def sayHello(self):
        print("大家好,我是", self.name, "我的score是" + str(self.score))


# 创建两个对象
a = Student(1, "张三", 20)
b = Student(2, "李四", 20)

# 给a对象动态添加sex属性
a.sex = "男"
a.sayHello()
print(a.sex)

# 没有给b添加sex属性，因此会报错
# print(b.sex)

# 私有属性被封装成  _类名__xxx 如__id 变成了_Student__id
# print(dir(a))
# print(dir(b))

print(a._Student__id)
print(b._Student__id)

# 对静态方法进行测试,可以通过对象也可以通过类调用
print("对静态方法进行测试")
Student.add(1, 2)
a.add(1,2)

# 对类方法进行测试,可以通过对象也可以通过类调用
print("对类方法进行测试")
Student.f()
a.f()

# 对@property 和 @name.setter测试
a.name = 200
print(a.name)
print(a._Student__name)
