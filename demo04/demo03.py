'''
数据库名称 ‘me’
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user` (
  `id` varchar(255) NOT NULL,
  `name` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES ('1', '鲁班七号', '123');
INSERT INTO `user` VALUES ('10', '张飞', '123');
INSERT INTO `user` VALUES ('2', '后裔', '123');
INSERT INTO `user` VALUES ('3', '阿珂', '123');
INSERT INTO `user` VALUES ('4', '马可波罗', '123');
INSERT INTO `user` VALUES ('5', '安琪拉', '123');
INSERT INTO `user` VALUES ('6', '妲己', '123');
INSERT INTO `user` VALUES ('7', '张良', '123');
INSERT INTO `user` VALUES ('8', '诸葛亮', '123');
INSERT INTO `user` VALUES ('9', '刘备', '123');

'''

import pymysql
import pymysql.cursors


# 主要对数据库的增删改查操作 更详细的API用法 请自己看文档


class User:

    def __init__(self, id, name, password):
        self.__id = id
        self.__name = name
        self.__password = password

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def password(self):
        return self.__password

    @password.setter
    def password(self, password):
        self.__password = password

    def __str__(self):
        return self.__id + "-" + self.__name + "-" + self.__password


# 数据库表结构和数据 见sample16文件夹

host = "localhost"
username = "root"
passwd = "lixin123"
port = 3306
charset = "utf8"
db = "me"


# <pymysql.connections.Connection object at 0x00000000025A39E8>
# print(connect)


# 测试select语句
def test_select():
    # 创建连接
    connect = pymysql.Connect(host=host, user=username, db=db, passwd=passwd, port=port, charset=charset)
    # 创建游标对象
    cursor = connect.cursor()

    # <pymysql.cursors.Cursor object at 0x0000000002A6BD68>
    # print(cursor)

    sql = "select * from user"

    users = []
    try:
        cursor.execute(sql)
        # (('1', '鲁班七号', '123'), ('10', '张飞', '123'), ('2', '后裔', '123'), ('3', '阿珂', '123'), ('4', '马可波罗', '123'), ('5', '安
        # 琪拉', '123'), ('6', '妲己', '123'), ('7', '张良', '123'), ('8', '诸葛亮', '123'), ('9', '刘备', '123'))

        # 返回的是元组列表
        data = cursor.fetchall()
        for user in data:
            temp = User(user[0], user[1], user[2])
            users.append(temp)

        for user in users:
            print(user)
        # print(data)
    except:
        connect.rollback()
    finally:
        cursor.close()
        connect.close()


def test_update():
    # 创建连接
    connect = pymysql.Connect(host=host, user=username, db=db, passwd=passwd, port=port, charset=charset)
    # 创建游标对象
    cursor = connect.cursor()
    sql = "update user set password = 123456 where id = 1"
    try:
        cursor.execute(sql)
        # 提交
        connect.commit()
        # 有数据返回的时候 才有数据 也就是查询的时候才会有
        data = cursor.fetchall()
        print(data)
    except:
        connect.rollback()
    finally:
        cursor.close()
        connect.close()


def test_insert():
    # 创建连接
    connect = pymysql.Connect(host=host, user=username, db=db, passwd=passwd, port=port, charset=charset)
    # 创建游标对象
    cursor = connect.cursor()
    sql = "insert into user values ('11','关羽','123456')"
    try:
        cursor.execute(sql)
        # 提交
        connect.commit()
        # 有数据返回的时候 才有数据 也就是查询的时候才会有
        # data = cursor.fetchall()
        # print(data)
    except:
        connect.rollback()
    finally:
        cursor.close()
        connect.close()


def test_delete():
    # 创建连接
    connect = pymysql.Connect(host=host, user=username, db=db, passwd=passwd, port=port, charset=charset)
    # 创建游标对象
    cursor = connect.cursor()
    sql = "delete from user where name = '关羽'"
    try:
        cursor.execute(sql)
        # 提交
        connect.commit()
        # 有数据返回的时候 才有数据 也就是查询的时候才会有
        # data = cursor.fetchall()
        # print(data)
    except:
        connect.rollback()
    finally:
        cursor.close()
        connect.close()

test_select()


