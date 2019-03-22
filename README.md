# 目录

## 强烈建议使用Anaconda搭建python环境（尤其是深度学习的同学）

### demo01 python基础知识

#### demo01.py

        一切都是对象，包括id、type、value(属性和方法)

        python中不可变对象有 数字、字符串、元组；可变的有列表、字典

#### demo02.py python字符串操作

        切片slice操作、len、join、split、input等

#### demo03.py python 常用数据结构 list、dict、tuple、set

        list 基本操作

            迭代: obj in list

            长度: len(list)

            增：append(obj)、extend(list)、insert(index,obj)

            删: remove(obj)、del list[index]、pop(index)

            查: index(obj) 如果找不到 会报错

        tuple 基本操作

            定义:(50,)、()

            tuple是不可变类型，删除的话只能把整个元组都删除，即 del tuple

        dict 基本操作

            key是否在dict中： key in dict

            取value(或赋值)：dict[key] 如果key不存在 会报错

            取value：dict.get(key) 如果key不存在 返回None

            删除key：dict.pop(key)、del dict[key]

            其他：clear(),has_key(key),items(),keys(),values(),update(dict2)

        set 基本操作

            定义：set(list),set(tuple),set()

            增：add(obj)

            删：remove(obj)

            交集：c.intersection(d)  c & d

            并集：c.union(d)  c \ d

            差集：c.difference(d)  c - d

            是否是子集：c.issubset(d) 

            是否是父集：c.issuperset(d) 

#### demo04.py for循环和可迭代对象

        可迭代对象：序列(列表、元组、字符串)、字典、iterator、generator、文件对象

        列表生成式 语法： [表达式 for item in 可迭代对象 if 判断条件]

        字典生成式 语法：{key:value for item in 可迭代对象 if 判断条件}

        集合生成式 语法： {表达式 for item in 可迭代对象 if 判断条件}

        元组没有生成式 只返回迭代对象 迭代对象只能用一次 第二次开始就没有数据了

        PS：for循环可以多个

#### demo05.py 函数相关

        函数名是个函数指针,可以想普通变量一样,赋值、修改等等

        函数参数可以接受函数（也就是C++的函数指针）

        参数传递传递的都是引用，只是有的类型是不可变类型，没法改

        浅拷贝，copy.copy() 拷贝父对象 不拷贝子对象

        深拷贝，copy.deepcopy() 完全拷贝父对象和子对象

        直接赋值，就是对象的引用

        参数：位置参数、命名参数、可变参数

            def f(a,b,c) 
                print(a,b,c)  

            位置参数：f(10,20,30) 输出 10 20 30

            命名参数：f(b = 30,c = 20,a = 10) 输出 10 30 20

            可变参数：*params,表示多个参数收集到一个元组中
                     **params，表示多个参数收集到一个字典中
        
        lambda表达式：只能写简单的，不如js好

            func = lambda a,b,c:a + b + c 返回值是 a+b+c
        
        nonlocal和global关键字：

            nonlocal关键字 用来声明外层变量(闭包的情况)

            global关键字 用来声明全局变量

#### demo06.py python内置函数第一波

        abs,divmod,int,ord,str,list,tuple,set,pow,sum,min,max,type,zip,reduce,map,filter,dir,input,type等
            
### demo02 面向对象，异常，IO

#### demo01.py 面向对象

        所有的类都继承object类，注意是小写的object

        __new__(self)方法用来创建对象，一般可以不用重写

        __del__(self)方法是析构函数

        __init__(self)方法用于初始化对象，一般做赋值操作

        __call__(self)方法是可以调用的方法

        __str__(self)方法是toString方法

        实例方法第一个参数必须是self

        实例属性在class内使用 self.属性名 的方式，在class外可以动态 修改和添加

        实例对象调用方法的本质：类名.方法名(实例对象)

            例子：
                a = Student() 
                a.sayHello()  

                等价于Student.sayHello(a)

        dir(obj)可以获取obj这个实例的所有属性

        isinstance(obj,class) 判断obj是不是class类型

        类属性需要写到字段中，访问需要用 类名.属性名 访问

        类方法需要注解@classmethod,第一个参数是cls, 类名.方法名 访问，其实也可以用实例访问

        静态方法需要注解@staticmethod,对参数没有要求, 类名.方法名 访问，其实也可以用实例访问

        类方法和静态方法不能访问实例属性和方法！！！否则会报错

        方法没有重载，会被覆盖

        私有属性：

            两个下划线开头的属性是私有属性

            class内部可以访问私有属性，外部不可以直接访问

            外部访问接口：_类名__属性名  PS：可以通过dir函数看到
        
        @property 是getter方法， @属性名.setter 是setter方法
            
            例子：
                @property
                def name(self):
                    return self.__name

                @name.setter
                def name(self,name):
                    self.__name = name

        继承允许多继承，但是及其不推荐

        继承时，子类的__init__方法中必须显式调用父类的__init__方法，python解释器不会自己调用

        可以通过 类名.mro() 查看类的继承层次结构

        子类调用父类的方法 用 super().方法名

#### demo02.py 特殊方法和运算符重载

        python的运算符实际上是调用对象的特殊方法实现的

            例子：c = a + b 其实是 c = a.__add__(b)

        特殊方法：

            __init__       初始化方法

            __del__        析构方法

            __str__        打印相关 如 print(a)

            __call__       能被调用 如 a()

            __getattr__    点运算符 如 a.xxx

            __setattr__    属性赋值 如 a.xxx = value

            __getitem__    索引运算 如 a[key]

            __setitem__    索引赋值 如 a[key] = value

            __len__        求长度   如 len(list)

        运算符对应的方法：记住英文缩写即可

            +:__add__         >,>=,!=:__gt__,__ge__,__ne__

            -:__sub__         <,<=,==:__lt__,__le__,__eq__

            *:__mul__

            /:__truediv__

            %:__mod__

            //:__floordiv__

            **:__pow__
            
       
        几个特殊属性：

            obj.__dict__

            obj.__class__   对象所属的类

            class.__bases__ 基类元组

            class.__base__  基类

            class.__mro__

            class.__subclasses__

#### demo03.py 异常处理

        导入了一个模块就相当于执行了一遍这模块的代码，类似于引入JavaScript文件

        __name__ 这个属性，如果这个python文件正在执行，那么这个脚本的__name__就是__main__,否则就是模块名

        所有的异常父类是BaseException

        处理语法： try..except..finally

        抛出异常:  raise BaseException()

#### demo04.py 文件处理相关

        os.path.exists(path)

        os.path.isfile(path)

        file = open(path,mode,encoding)

            读：file.read() file.readlines() file.readline()
            PS: 连带换行符都读进来了，需要手动去掉

            写：file.write() file.writelines()

            关: file.close()
        
        文件操作模块：shutil 和 os 

### demo03 正则表达式，多进程，简单爬虫

#### demo01.py 正则表达式 简单爬虫案例

        贪婪和非贪婪：

            正则表达式 ab* 如果查找 "abbbc" 将找到abbbb;如果改成非贪婪，即 ab*? 将找到 a

        反斜杠问题：

            正则表达式里面有个转义字符，如果要表达两个反斜杠，就得携程 "\\\\"

            为了解决这个问题，用python的原生字符串，即 r"\\"

        urllib(python3!!!)：

            def getHTML(url,charset="utf-8"):
                return urllib.request.urlopen(url).read().decode(charset)

            def download(url,filePath):
                urllib.request.urlretrieve(url,filePath)

        re模块：

            pattern = re.compile(patternString)

            match = re.match(pattern,string,flags) 返回结果或None

            match对象包含我们需要的全部信息

#### demo02.py 多进程和多线程

        多进程模块：multiprocessing

            1、Process([group [, target [, name [, args [, kwargs]]]]])

                target：表示这个进程实例所调用对象,就是执行哪个函数

                args：表示调用对象的位置参数元组，就是函数参数列表

                kwargs：表示调用对象的关键字参数字典，也是函数参数列表，只不过是字典

                name：为当前进程实例的别名；
                
                group：大多数情况下用不到；
            
            2、Process常用方法和属性：

                is_alive()：判断进程实例是否还在执行

                join([timeout])：是否等待进程实例执行结束，或等待多少秒

                start()：启动进程实例（创建子进程）

                run()：如果没有给定target参数，对这个对象调用start()方法时，就将执行对象中的run()方法

                terminate()：不管任务是否完成，立即终止
            
                name：当前进程实例别名，默认为Process-N，N为从1开始递增的整数

                pid：当前进程实例的PID值

            3、Pool 进程池

                pool = multiprocessing.Pool(number)

                pool.apply_async(执行体,参数元组)
    
                pool.close()

                pool.join()
            
            4、线程(threading模块 写法和Process一样)

            5、线程池(threadpool模块 需要自己安装, 用法同进程池)

                pip install threadpool

#### demo03.py 用多进程 和 正则表达式 爬虫校花网图片封面

        只针对没有设置反爬虫的网站，只用正则表达式解析html标签 2018-10-26 测试没有问题

        是 demo10 和 demo11 的整合

        注意：爬虫的文件夹需要提前建立好，我没有加判断

        为什么不用多线程而用多进程？

            因为python锁的问题，线程进行锁竞争、切换线程会消耗资源；而进程中的锁是独立的

### demo04 生成器 迭代器 装饰器 常用模块 mysql交互

#### demo01.py

        在前面已经学习了生成器：

            列表生成式 语法： [表达式 for item in 可迭代对象 if 判断条件]

            字典生成式 语法：{key:value for item in 可迭代对象 if 判断条件}

            集合生成式 语法： {表达式 for item in 可迭代对象 if 判断条件}

        迭代器：

            能用for循环的对象都是可迭代对象，即collections.Iterable 

            生成器都是 Iterator 对象，但 list 、 dict 、 str 虽然是 Iterable ，却不是 Iterator 。把 list 、 dict 、 str 等 Iterable 变成 Iterator 可以使用 iter() 函数

            凡是可作用于 for 循环的对象都是 Iterable 类型

            凡是可作用于 next() 函数的对象都是 Iterator 类型

        装饰器：

            类似于Java里面的注解，在设计模式上和装饰者模式有异曲同工之妙

            实际上是通过闭包的方式，增强原函数的功能

            装饰器通俗点就是说，给函数增加一层层的包装

            例子：
                # @wrapper的过程就是 f = wrapper(f),当f本身是个函数，所以f可以执行
                @wrapper
                def f():
                    print("这里是f")

#### demo02.py

        一个python文件就是一个模块，比如a.py，那么a就是一个模块

        为了解决模块重名的问题，引入了包，一个包就是一个文件夹，引入了包的模块，模块名就行 包名.模块名

        目录结构：包下面必须有一个__init__.py的文件，包里面可以放其他包和模块

        __init__.py文件是将一个文件夹变成包,该文件中可以定义__all__ 列表，该列表定义了能被外部访问的项

        导入这个模块就相当于执行了这个模块的代码

        导入方式：

            import 包名.模块名 

            from 包名.模块名 import 具体项

#### demo03.py python和mysql交互

        python3用pymysql模块，python2用mysqldb模块

        安装方式： pip install pymysql 

        pycharm的安装方式：打开setting--找到当前项目--找到Project Interpreter--点击加号--输入模块

        检验安装是否成功：在cmd python环境下 import pymysql不报错则安装成功

        步骤：创建连接-创建游标-写sql语句-执行(增删改还需要commit)-异常处理-关闭游标和关闭连接
            
            connect = pymysql.Connect(host=host,user=user,db=db,passwd=passwd,port=port,charset=charset)

            cursor = connect.cursor()
            
            sql = "update user set password = 123456 where id = 1"

            try:

                cursor.execute(sql)

                connect.commit()

                data = cursor.fetchall()

            except:

                connect.rollback()

            finally:

                cursor.close()

                connect.close()

### demo05 numpy,matplotlib,scipy,pandas 机器学习相关模块


#### demo01.py numpy 模块基本操作

        这里涉及到许多线性代数的专业术语，如果有不懂的术语，请参考线性代数
        
        numpy库 不是python内置模块，需要自行安装，如果python是Anaconda安装的，那么这个模块已经安装上了

        numpy主要处理高维数组的库，什么东西需要用到高维数组？图像算一个，一幅图像有那么多个像素点，每个像素点由RGB构成

        numpy中的数组类ndarray和python标准库中的array.array不一样，后者只处理一维数组和提供少量功能

        numpy.ndarray基本属性：

            ndim：数组的维度(矩阵的秩))
                
                这个维度和数学上的维度还有些差异，可以简单的理解为有几层括号，维度就是几

                arr1 = [1,2,3,4,5,6] 维度是1

                arr2 = [[1,2,3],[4,5,6]] 维度是2

                arr3 = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]] 维度是3
                

            shape:返回值是元组，元组中的元素代表每一维度的长度，

                arr1 = [1,2,3,4,5,6] shape = (6,)

                arr2 = [[1,2,3],[4,5,6]] shape = (2,3)

                arr3 = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]] shape = (2,2,3)

                元组每个数代表最外层括号有几个元素，次外层有几个元素，次次外层有几个元素....

            size：数组元素总个数

            dtype：(data type)数组元素对象的类型，可以使用python标准类型，也可以使用其他类型

            itemsize：数组中每个元素的字节大小

            reshape：重构数组shape,按照指定的shape重新排列数组，如果无法重构则报错

        numpy.ndarray基本方法：

            astype: 转换数组的数据类型,返回值是新数组

            min,max,sum:最小值，最大值，元素求和；axis参数你可以吧运算应用到数组指定的轴上

        numpy模块中的方法：

            array：构造函数，参数有数组数据和dtype

            asarray: 将能转换成ndarray的东西 转成ndarray

            arange: 同range(start,end,step) 左闭右开，只是后者返回的是迭代器，前者是数组(ndarray)

            zeros: zeros(size)  zeros(row,col) 全是0的数组

            empty: 用法同zeros，只是初始值是随机值

            eye：eye(number) 创建number * number 的单位数组

            full：full(shape,full_value) 所有值都一样的数组

            random.rand(shape) 根据shape生成0-1随机数

            random.uniform(0, 100) 产生一个随机数

            random.randint(0, 100) 产生一个随机整数

            linspace：linspace(开始值，终值，元素个数) 默认包含终值

            dot：用于矩阵乘法

            amax(ndarray,axis=0/1) :指定轴(0列1行)的最大值

            amin,mean,amax,std用法一致，分别是求最小，均值，最大，方差

            vstack((arr1,arr2,)),hstack((arr1,arr2)) 数组的垂直拼接,水平拼接

            np.c_[array1,array2] 表示按列向右扩展

            np.r_[array1,array2] 表示按行向下扩展

            genfromtxt(filePath,delimiter) 读取csv文件 返回ndarray
        
        numpy.ndarray运算：

            算数运算符 比较运算符：按照对应元素进行加减乘除

            矩阵乘法：numpy.dot(A,B)


#### demo02.py numpy 切片

        切片都是左闭右开

        切片可以接条件表达式

        一维数组： 和 列表一样

        二维数组： arr[行切片，列切片]

                其中每一行和每一列都是一个一维数组

        三维数组：比如RGB图像，[[[R,G,B],[R,G,B],[R,G,B],[R,G,B]],[],[],[].....]

                arr[最外层数组切片,二维数组切片,一维数组切片]

                其实就是arr[RGB图片的第几行像素,这一行像素的前几个像素,这个像素的哪几个值]


#### demo03.py matplotlib 绘图模块

        安装：如果是Anaconda 那么已经安装了，否则需要 pip install matplotlib 导入不报错则安装成功，如果是pycharm还是从setting里面安装

        散点图：scatter

        饼图：pie

        折线图：plot

        子图：subplot

        柱状图：bar

        直方图：hist

#### demo04.py pandas 模块

        数据结构：

            Series：一维数组

            Time-Series：以时间为索引的Series。

            DataFrame：二维的表格型数据结构

            Panel ：三维的数组，可以理解为DataFrame的容器。

### demo06 tensorflow 框架

        demo01.py tensorflow的基本使用,并用tensorflow进行回归

        demo02.py 初始mnist数据集，调整神经网络层数，参数，激活函数，变量初始化方式

        demo03.py 深入mnist数据集，用DNN识别mnist数据集，并用matplotlib做可视化

        demo04.py tensorflow保存模型和加载模型

        demo05.py mnist模型加载并测试

        demo06.py 加入了dropout层防止过拟合，使用交叉熵函数作为loss函数

