# coding = utf-8
import os


class FileUtils:

    # 读取文件，并打印出来
    def readFile(self, filePath):
        if (os.path.exists(filePath)):
            if (os.path.isfile(filePath)):
                fopen = open(filePath, 'r')
                return fopen.read()
            else:
                print(filePath + "不是一个文件")
        else:
            print(filePath + "路径不存在")

    # 文件复制
    def copy(self, src, dest):
        if (os.path.exists(src)):
            if (os.path.isfile(src)):
                srcFile = open(src, 'r')
                destFile = open(dest, 'w')
                content = srcFile.read()
                destFile.write(content)

                srcFile.close()
                destFile.close()
            else:
                print("源路径不是文件")
        else:
            print("路径不存在")


def test_os():
    workPath = os.getcwd()
    print(workPath)

    # 获取当前所有文件和目录 os.listdir(path)

    path = 'd:/'
    files = os.listdir(path)
    print(len(files))
    print(files)

    # 删除一个文件 os.remove(path)
    fpath = path + 'sfdsf.jpg'
    if (os.path.exists(fpath)):
        print(os.path.isfile(fpath))
        os.remove(fpath)
        print(os.path.exists(fpath))

    # 返回一个路径的目录名和文件名:os.path.split(path)
    file = open('d:/1.txt', 'w')
    filepath = 'd:/1.txt'
    list = os.path.split(filepath)
    print(list)  # ('d:/','1.txt')

    # 分离扩展名：os.path.splitext()
    print(os.path.splitext(filepath))  # ('d:/1','.txt')
    print('后缀类型为' + os.path.splitext(filepath)[1][1:])

    # 获取路径名os.path.dirname(path)
    # 获取文件名os.path.basename(path)
    dir = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    print(dir)
    print(base)

    # 创建多级目录：os.makedirs(path)
    path = 'D:\\myself\\python\\docment\\2.jpg'

    if (os.path.exists(path)):
        pass
    else:
        os.makedirs(path)

    # 获取文件大小：os.path.getsize(filename)字节
    print(os.path.getsize(path))


# 测试shutil模块
def shutil_test():
    # 导入shutil模块和os模块
    import shutil, os

    # 复制单个文件
    shutil.copy("C:\\a\\1.txt", "C:\\b")
    # 复制并重命名新文件
    shutil.copy("C:\\a\\2.txt", "C:\\b\\121.txt")
    # 复制整个目录(备份)
    shutil.copytree("C:\\a", "C:\\b\\new_a")

    # 删除文件
    os.unlink("C:\\b\\1.txt")
    os.unlink("C:\\b\\121.txt")
    # 删除空文件夹
    try:
        os.rmdir("C:\\b\\new_a")
    except Exception as ex:
        print("错误信息：" + str(ex))  # 提示：错误信息，目录不是空的
    # 删除文件夹及内容
    shutil.rmtree("C:\\b\\new_a")

    # 移动文件
    shutil.move("C:\\a\\1.txt", "C:\\b")
    # 移动文件夹
    shutil.move("C:\\a\\c", "C:\\b")

    # 重命名文件
    shutil.move("C:\\a\\2.txt", "C:\\a\\new2.txt")
    # 重命名文件夹
    shutil.move("C:\\a\\d", "C:\\a\\new_d")


def main():
    test_os()
    src = "samples/files/1.txt"
    dest = "samples/files/2.txt"
    fileUtil = FileUtils()
    fileUtil.copy(src, dest)


if __name__ == '__main__':
    main()
