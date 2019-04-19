import os
from PIL import Image

def image_to_byte(fileInput, width, height):
    '''
    desc:
        将图片改变尺寸后转成字节后返回
    params:
        fileInput:图片输入路径
        width: 输出图片宽度
        height: 输出图片高度
    return:
        图片字节
    '''
    img = Image.open(fileInput)
    img = img.resize((width,height),Image.ANTIALIAS)
    return img.tobytes()

def resize_image(fileInput, fileOutput, width, height, type="png"):
    '''
    desc：
        将图片改变尺寸后保存成新图片
    params：
        fileInput:图片输入路径
        fileOutput：图片输出路径
        width: 输出图片宽度
        height: 输出图片高度
        type:输出文件类型，png,gif,jpeg
    '''
    img = Image.open(fileInput)
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    newImg = img.resize((width, height), Image.ANTIALIAS)
    newImg.save(fileOutput)
    print(str(fileInput) + " has been resized successfully!")


def resize_dir_image(dir_path, width, height, type="png"):
    '''
    desc：
        将图片文件夹里面所有的图片改变尺寸
    params：
        dir_path: 图片文件夹
        width: 输出图片宽度
        height: 输出图片高度
        type:输出文件类型，png,gif,jpeg
    '''
    for img_name in os.listdir(dir_path):
        path = dir_path + "/" + img_name
        resize_image(path, path, width, height, type)
    print("the directory %s has been resized successfully" % (dir_path))
