# 用多进程和正则表达式爬虫

import re
import urllib
import urllib.request
import multiprocessing
import os


# 根据url 和 charset 获取html字符串
# 如果有反爬虫 不好用了 还需要添加别的 以后再说
def getHTML(url, charset="utf-8"):
    return urllib.request.urlopen(url).read().decode(charset)


# 下载
def download(url, filePath):
    print(url, "开始下载")
    urllib.request.urlretrieve(url, filePath)
    print(url, "开始完成")


# 多进程下载校花网第0页图片封面
def downLoad_0_page():



    imageIndex = 0
    pageNumber = 0
    root = "http://www.xiaohuar.com"
    pageURL = "http://www.xiaohuar.com/list-1-" + str(pageNumber) + ".html"
    html = getHTML(pageURL, "gbk")
    # 图片链接有两类
    # https://wx.dxs6.cn/api/xiaohua/upload/min_img/20180909/20180909NgpOY4F7jU.jpg"分析url
    # /d/file/20180905/3e432df2173ac4f25ffb14b6f94e1bc7.jpg 分析url
    reg = r"https://wx.dxs6.cn/api/xiaohua/upload/min_img/.+?jpg"
    imgre1 = re.compile(reg)
    reg = r"/d/file/.*?jpg"
    imgre2 = re.compile(reg)

    # 获得第一类图片url 就是完整路径
    list = re.findall(imgre1, html)
    # 获得第二类图片url 不是完整路径
    list2 = re.findall(imgre2, html)

    for index in range(0, len(list2)):
        list2[index] = root + list2[index]

    list.extend(list2)

    # 25条数据 正好一页25个图片封面
    # print(len(list))

    # 创建进程池 里面有3个进程
    pool = multiprocessing.Pool(3)

    try:
        for url in list:
            # 注意 这里的文件夹需要创建好 我没有加文件夹的判断
            filePath = "D:/images/" + str(imageIndex) + ".jpg"
            pool.apply_async(download, (url, filePath,))
            imageIndex += 1
    except:
        print("出现错误了")
    finally:
        pool.close()
        pool.join()

    print("爬虫完毕")


# 爬第0-number页的图片封面
def downLoad_n_page(number=0):
    if (os.path.exists("D:/images/")):
        pass
    else:
        os.makedirs("D:/images/")
    imageIndex = 0
    pageNumber = 0
    root = "http://www.xiaohuar.com"
    # 创建进程池 里面有5个进程
    pool = multiprocessing.Pool(5)
    while (pageNumber <= number):
        pageURL = "http://www.xiaohuar.com/list-1-" + str(pageNumber) + ".html"
        html = getHTML(pageURL, "gbk")
        # 图片链接有两类
        # https://wx.dxs6.cn/api/xiaohua/upload/min_img/20180909/20180909NgpOY4F7jU.jpg"分析url
        # /d/file/20180905/3e432df2173ac4f25ffb14b6f94e1bc7.jpg 分析url
        reg = r"https://wx.dxs6.cn/api/xiaohua/upload/min_img/.+?jpg"
        imgre1 = re.compile(reg)
        reg = r"/d/file/.*?jpg"
        imgre2 = re.compile(reg)

        # 获得第一类图片url 就是完整路径
        list = re.findall(imgre1, html)
        # 获得第二类图片url 不是完整路径
        list2 = re.findall(imgre2, html)

        for index in range(0, len(list2)):
            list2[index] = root + list2[index]
        list.extend(list2)

        try:
            for url in list:
                filePath = "D:/images/" + str(imageIndex) + ".jpg"
                pool.apply_async(download, (url, filePath,))
                imageIndex += 1
        except:
            print("出现错误了")

        pageNumber += 1

    pool.close()
    pool.join()
    print("爬虫完毕")


def main():
    downLoad_n_page(10)


if __name__ == '__main__':
    main()
