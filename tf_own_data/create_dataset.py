import ImageResizeTool
import os
import tensorflow as tf

# 一共多少类
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

'''
TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议内存块包含了字段 Features)。我们可以写一段代码获取你的数据， 将数据填入到Example协议内存块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过tf.python_io.TFRecordWriter 写入到TFRecords文件。
'''


def create_record(root_dir, output_name="train.tfrecords"):
    '''
    desc:
        生成tfrecords文件
    params:
        rood_dir:图片根文件夹路径
        output_name:要生成数据集文件的名字
    '''
    writer = tf.python_io.TFRecordWriter(output_name)
    for index, name in enumerate(classes):
        image_dir_path = root_dir + "/" + name
        for image_name in os.listdir(image_dir_path):
            image_path = image_dir_path + "/" + image_name
            img_raw = ImageResizeTool.image_to_byte(image_path, 128, 128)

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(filename):
    '''
    desc:
        读取二进制数据
    params：
        filename：tfrecords文件路径
    return:
        image,label
    '''
    # 创建文件名队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    # reshape为128*128的3通道图片
    img = tf.reshape(img, [128, 128, 3])
    # 在流中抛出img张量
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # 在流中抛出label张量
    label = tf.cast(label, tf.int32)
    return img, label

if __name__ == "__main__":
    with tf.Session() as sess:
        print(sess.run(read_and_decode("train.tfrecords")))

