class Person:
    def __init__(self, id, name, password):
        self.__id = id
        self.name = name
        self.password = password

    def __str__(self):
        return "[id:%s,name:%s,password:%s]" % (self.__id, self.name, self.password)

    def __call__(self):
        print("我虽然是个class，但是我可以被调用")

    # 相当于定义了一个属性叫id，这个id属性是__id的只读属性，当id改变时，__id也改变
    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    def __lt__(self, student):
        if (self.__id < student.id):
            return True
        else:
            return False

    def __gt__(self, student):
        if (self.__id > student.id):
            return True
        else:
            return False


s1 = Person(1, "小明", 123)
s2 = Person(2, "小红", 123)

# s1.id = 20

print(s1)
print(s2)
print(s1 < s2)
print(s1 > s2)
s1()
