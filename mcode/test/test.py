# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 9/26/18 8:00 PM


import os
import sys
import cv2  # imread
import torch
import numpy as np
import json
from os.path import realpath, dirname, join
import xml.dom.minidom as minidom
from mcode.net import SiamRPNBIG
from mcode.run_SiamRPN import SiamRPN_init, SiamRPN_track
from mcode.utils import get_axis_aligned_bbox, cxy_wh_2_rect

def write_res_box(res_path, default='img_siam', c_type='front_foot_left', res=None, begin_index=0):
    if not res:
        res_file = os.path.join(res_path, c_type + ".json")
        if not os.path.exists(res_file):
            print("No any res for " + c_type)
            return

        with open(res_file, 'r') as f:
            string = f.read()
            j = json.loads(string)
    else:
        j = res

    ori_img = res_path.replace("Omar_track", "Omar_res")
    res_img = os.path.join(res_path, default)
    if not os.path.exists(res_img):
        os.makedirs(res_img)
    bbox = j['res']
    if len(bbox) > 500:
        bbox = bbox[:500]
    for index, item in enumerate(bbox):
        ind = index + 1 + begin_index
        img_name = "%09d.jpg" % ind
        img_path = os.path.join(ori_img, img_name)
        out_path = os.path.join(res_img, img_name)
        img = cv2.imread(img_path)
        x = int(item[0])
        y = int(item[1])
        w = int(item[2])
        h = int(item[3])
        if x <= 0 :
            x = 2
        if y <= 0:
            y = 2

        if len(item) > 4:
            state = int(item[4])
        else:
            state = int(1)
        if state == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 3)
        elif state == 3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        elif state == 4:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        elif state == 5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imwrite(out_path, img)

        # if index >= 500:
        #     print("done 500")
    print("done save")

def read_xml(filename, c_type):
    # print 'Loading: {}'.format(filename)
    num_classes = 5
    _classes = ('__background__',  # always index 0
                'front_foot_right', 'front_foot_left', 'back_foot_right', 'back_foot_left')
    # _class_to_ind = dict(zip(_classes, range(num_classes)))

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin'))
        y1 = float(get_data_from_tag(obj, 'ymin'))
        x2 = float(get_data_from_tag(obj, 'xmax'))
        y2 = float(get_data_from_tag(obj, 'ymax'))
        class_name = str(get_data_from_tag(obj, "name")).lower().strip()
        if class_name == c_type:
            boxes = [x1, y1, x2 - x1, y2 - y1]
            return boxes

    return [0.0, 0.0, 0.0, 0.0]

def read_txt(filename, c_type):
    boxes = [0.0, 0.0, 0.0, 0.0]
    with open(filename, 'r') as f:
        for init_rect in f:
            init_rect = init_rect[:-1]
            coord = init_rect.split(' ')
            if coord[-1] == c_type:
                boxes = [float(i) for i in coord[:-1]]

    return boxes

def load_seq(seq_path, bbox_path, prefix="", c_type='front_foot_left'):
    seq = {}
    seq_list = [ os.path.join(prefix, seq_i) for seq_i in seq_path]

    # # TODO: only 300 seqs handle
    if len(seq_list) > 500:
        seq_list = seq_list[:500]

    s_path = bbox_path.split('/')
    t_path = '/'.join(s_path[:-1])

    number = int(s_path[-1].split('.')[0])
    while not os.path.exists(bbox_path + ".xml") or not os.path.exists(bbox_path + ".txt"):
        number += 1
        bbox_path = os.path.join(t_path, ("%09d" % number))

    print(bbox_path + " has " + ("%09d.xml" % (number)))

    if os.path.exists(bbox_path + ".xml"):
        bbox_path = bbox_path + ".xml"
    else:
        bbox_path = bbox_path + ".txt"

    seq['s_frames'] = seq_list[number-1:]
    seq['len'] = len(seq_path)

    if ".txt" in bbox_path:
        init_rect = read_txt(bbox_path, c_type)
    else:
        init_rect = read_xml(bbox_path, c_type)

    seq['init_rect'] = init_rect

    return seq, number - 1

# load net
net_file = join('/home/slh/torch/SiamTracker/mcode/test', 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()


# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

TARGET_PATH = "/datasets/MLG/distance/Omar_res"
TRACKDIR = "/home/slh/tf-project/mouse/MDNet"
C_TYPE='front_foot_right'
res_file_list = os.listdir(TARGET_PATH)
for it in res_file_list:
    sec_path = os.path.join(TARGET_PATH, it)
    res_path = sec_path + "/resources.txt"
    if os.path.exists(res_path) :
        with open(res_path, 'r') as f:
            for line in f:
                sp = line[:-1].split(" ")
                seq_path_tmp = sp[0]

                if not 'Dat1c2Right_2018-03-29-184051' in seq_path_tmp:
                    continue

                box_path_tmp = seq_path_tmp.replace('Omar_res', 'Omar_track')
                box_path = os.path.join(box_path_tmp, "xml")
                tmp_list = os.listdir(box_path)
                if len(tmp_list) > 0:
                    seq, begin_index = load_seq(os.listdir(seq_path_tmp), os.path.join(box_path, ("%09d" % (1))),
                                                prefix=seq_path_tmp, c_type=C_TYPE)
                else:
                    print("Not has results.")
                    continue
                rp = "./"
                save_img = False
                if not os.path.exists(os.path.join(box_path_tmp, C_TYPE + ".json")):
                    init_bbox = seq['init_rect']
                    res_dict = {}
                    res_dict['res'] = [init_bbox]
                    target_pos = np.array([init_bbox[0] + 0.5 * init_bbox[2], init_bbox[1] + 0.5 * init_bbox[3]])
                    target_sz = np.array(init_bbox[2:])
                    print(target_pos)
                    print(target_sz)
                    im = cv2.imread(seq['s_frames'][0])
                    state = SiamRPN_init(im, target_pos, target_sz, net)
                    for index, item in enumerate(seq['s_frames'][1:]):
                        im = cv2.imread(item)
                        state = SiamRPN_track(state, im)  # track
                        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                        res_dict['res'].append(res)
                        print(item, end=' ')
                        print(res, end=' ')
                        print(state['score'])
                        # write _res_box(box_path_tmp, res=res, begin_index=begin_index)

                    write_res_box(box_path_tmp, res=res_dict, begin_index=begin_index)