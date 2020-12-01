# -*- coding: utf-8 -*-

import os
from glob import glob
from xml.etree import ElementTree
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--annotation-root', help='Annotation root directory')
    parser.add_argument('--label-names', default='label.names', help='label.names filename')
    args = parser.parse_args()

    # fetch XML file names
    xml_file_list = glob(args.annotation_root + os.sep + '*.xml')

    # Check if previous label names file exists
    if not os.path.isfile(args.label_names):
        label_map = {}
        # also, search over annotations and create label list
        for xml_filename in tqdm(xml_file_list, desc='(1/2) 레이블 탐색 작업'):
            with open(xml_filename, 'r') as f:
                xml_body = f.read()
                root = ElementTree.fromstring(xml_body)
                for item in root.findall('object'):
                    label_map[item.find('name').text] = True

        label_list = list(label_map.keys())
        label_list.sort()
        label_count = len(label_list)

        # Save label list as txt file for further use
        with open(args.label_names, 'w') as f:
            for label in label_list:
                f.write(label + '\n')  # Unix-like formatting
            f.flush()
    else:
        label_list = list()
        with open(args.label_names, 'r') as f:
            for label in tqdm(f, desc='(1/2) 레이블 읽기 작업'):
                label_list.append(label.rstrip())
        label_count = len(label_list)

    PARSED_ANNOTATION_DIR = '.' + os.sep + '.annotations'
    if not os.path.exists(PARSED_ANNOTATION_DIR):
        os.mkdir(PARSED_ANNOTATION_DIR)

    for xml_filename in tqdm(xml_file_list, desc='(2/2) 레이블 매칭 및 신뢰도 작업'):
        with open(xml_filename, 'r') as f:
            xml_body = f.read()

            # Parse XML file
            root = ElementTree.fromstring(xml_body)
            image_path = os.sep.join(xml_filename.split(os.sep)[:-2]) + os.sep + 'JPEGImages' + os.sep + root.find(
                'filename').text
            image_width =int(root.find('size').find('width').text)
            image_height = int(root.find('size').find('height').text)

            object_list = root.findall('object')
            object_label_list = [item.find('name').text for item in object_list]
            xyminmax_list = [item.find('bndbox') for item in object_list]
            xyminmax_list = [
                [int(float(x)) for x in [obj.find('xmin').text, obj.find('xmax').text, obj.find('ymin').text, obj.find('ymax').text]]
                for obj in xyminmax_list]

            xycenter_wh_list = [
                # Effecitively each are (x_center, y_center, width, height)
                ((obj[0] + obj[1]) / 2, (obj[2] + obj[3]) / 2, obj[1] - obj[0], obj[3] - obj[2])
                for obj in xyminmax_list
            ]

            xycenter_wh_list_norm = [
                # Effectively each are normalized to (0, 1]. center xy position are also normalized
                (obj[0] / image_width, obj[1] / image_height, obj[2] / image_width, obj[3] / image_height)
                for obj in xycenter_wh_list
            ]

            object_list_len = len(object_list)

            xml_filename_nopath = xml_filename.split(os.sep)[-1]
            with open(PARSED_ANNOTATION_DIR + os.sep + xml_filename_nopath.replace('.xml', '.anot'), 'w') as wf:
                wf.write(image_path + '\n')
                wf.write(f'{image_width} {image_height}\n')
                for object_idx in range(object_list_len):
                    object_name = object_label_list[object_idx]
                    xycenter_wh_norm = list(xycenter_wh_list_norm[object_idx])

                    object_one_hot_list = [0.0] * label_count
                    object_one_hot_list[label_list.index(object_name)] = 1.

                    wf.write(' '.join([str(x) for x in xycenter_wh_norm]) + ' ')
                    wf.write(' '.join([str(x) for x in object_one_hot_list]) + '\n')
                wf.flush()