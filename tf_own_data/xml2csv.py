
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

path = "d:/lixin/data/test"


def xml_to_csv(path, csv_file_path="file.csv"):
    '''
    desc:
        xml转csv文件
    params：
        path：xml所在文件夹路径
        csv_file_path: 生成csv文件的路径
    returns:
        csv文件
    '''
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(csv_file_path, index=None)
    print('Successfully converted xml to csv.')
    return xml_df


def main():
    xml_to_csv(path)


if __name__ == "__main__":
    main()
