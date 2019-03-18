# python 字符串操作

str = "1q2w3e4r5t"

# slice 切片操作
print("str[:5]=", str[:5])  # 从头开始到下标为5结束，如果超过length，就到最后
print("str[1:3]=", str[1:3])  # 从下标为1 到下标为3 左闭右开区间
print("str[:-1]=", str[:-1])  # -1表示从逆序第一个
print("str[4:-1]=", str[4:-1])  # 3e4r5
print("str[:]=", str[:])  # 全部

# str 也是不可变类型
# 36218032
print(id(str))
str = str + "fds"
# 36223792
print(id(str))
print(str)


str1 = input("请输入一个字符串")
print(str1)
print(len(str1))

array = ["12","df"]
print("".join(array))
print("".join(array).split("2"))

