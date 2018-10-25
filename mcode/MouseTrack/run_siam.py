# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 9/27/18 3:09 PM

import os
import sys
import torch
import numpy as np
import cv2
from mcode.MouseTrack.utils.tools import judge_detect_res
from mcode.net import SiamRPNBIG
from mcode.run_SiamRPN import SiamRPN_init, SiamRPN_track
from mcode.utils import get_axis_aligned_bbox, cxy_wh_2_rect


def run_SiamRPN(detector, seq, gpu_number='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    # network start
    __prefix_path__, _ = os.path.split(os.path.realpath(__file__))
    net_file = os.path.join(__prefix_path__, 'utils/model/siam/SiamRPNBIG.model')
    net = SiamRPNBIG()
    net.load_state_dict(torch.load(net_file))
    net.eval().cuda()

    # warm up
    for i in range(10):
        net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
        net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

    # init bbox
    track_type = seq['type']
    init_bbox = seq['init_rect']
    res_dict = {}
    res_dict['res'] = [init_bbox]
    target_pos = np.array([init_bbox[0] + 0.5 * init_bbox[2], init_bbox[1] + 0.5 * init_bbox[3]])
    target_sz = np.array(init_bbox[2:])
    print(target_pos)
    print(target_sz)
    im = cv2.imread(seq['s_frames'][0])
    state = SiamRPN_init(im, target_pos, target_sz, net)

    sequences_list = seq['s_frames'][0:]
    for index, item in enumerate(sequences_list[1:]):
        im = cv2.imread(item)
        # state['target_pos'] = target_pos
        # state['target_sz'] = target_sz
        # state['score'] = score
        state = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        if state['score'] < 0.3:
            detect_res, sit = judge_detect_res(detector, res_dict['res'][-1], sequences_list[index:], track_type)
            print("Detect res: ", end='')
            print(detect_res)
        res_dict['res'].append(res)

        print(item, end=' ')
        print(res, end=' ')
        print(state['score'])
        # write _res_box(box_path_tmp, res=res, begin_index=begin_index)

    res['res'] = tmp # scripts.butil.matlab_double_to_py_float(res['res'])
    # m.quit()
    return res

