import re
import urllib
import urllib.request


def hello_world():
    # 在起始位置匹配
    print(re.match('www', 'www.runoob.com').span())  # (0,3)
    # 不在起始位置匹配
    print(re.match('com', 'www.runoob.com'))  # None


# 根据url 和 charset 获取html字符串
# 如果有反爬虫 不好用了 还需要添加别的 以后再说
def getHTML(url, charset="utf-8"):
    return urllib.request.urlopen(url).read().decode(charset)


# 下载
def download(url, filePath):
    urllib.request.urlretrieve(url, filePath)


# 校花网图片下载demo，该网站没有设置反爬虫策略，可以直接解析HTML
def test_download_img(charset='utf-8'):
    url = "http://www.xiaohuar.com/news-1-134.html"
    html = getHTML(url, "gbk")
    reg = r"/d/file/\d+?.jpg"
    imgre = re.compile(reg)
    imglist = re.findall(imgre, html)
    print(imglist)
    root = "http://www.xiaohuar.com"
    x = 0
    for imgurl in imglist:
        download(root + imgurl, "d:/yuzhu/" + str(x) + ".jpg")
        x += 1


def reg_test():
    # 判断字符串是否全是小写字母
    s1 = 'adkkdk'
    s2 = 'abc123efg'
    an = re.search("^[a-z]+?$", s1)
    if (an):
        print("s1:", an.group(), "全是小写")
    else:
        print("不全是小写")

    # group 函数
    # 0 表示所有 1-n 表示第 1-n 个括号，没有括号1-n则报错
    a = "123abc456"
    print(re.search("([0-9]*)([a-z]*)([0-9]*)", a).group(0))  # 123abc456,返回整体
    print(re.search("([0-9]*)([a-z]*)([0-9]*)", a).group(1))  # 123
    print(re.search("([0-9]*)([a-z]*)([0-9]*)", a).group(2))  # abc
    print(re.search("([0-9]*)([a-z]*)([0-9]*)", a).group(3))  # 456
    # 邮箱测试
    print(re.match(r"^\d+?@qq.com", "90703518@qq.com").group())
    print(re.match(r"^\w+?@\w+.\w+$", "90703518@qq.com").group())

    # url测试
    url = "http://www.xiaohuar.com/news-1-134.html"
    html = getHTML(url, "gbk")
    # 本来想写 ^/d/file/\d+?.jpg$ 可是findall方法不给力 不匹配 不知道是不是我的问题
    reg = r"/d/file/\d+?.jpg"
    imglist = re.findall(reg, html)
    # imglist = ['/d/file/20140916020044188.jpg', '/d/file/20140916020044188.jpg', '/d/file/20140916020045170.jpg', '/d/file/20140916020045172.jpg', '/d/file/20140916020046123.jpg']
    for imgurl in imglist:
        print(re.match(reg, imgurl).group())


def main():
    reg_test()


if __name__ == '__main__':
    main()
