# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 10/15/18 2:45 PM

import os
import torch
import cv2
import sys
import numpy as np
from mcode.MouseTrack.utils.tools import drawBox, init_net, init_track, save_pictures, rect_box
from mcode.run_SiamRPN import SiamRPN_track
from mcode.utils import cxy_wh_2_rect

os.environ['CUDA_VISIBLE_DEVICES'] = '11,12'

__prefix_path__, _ = os.path.split(os.path.realpath(__file__))
video_name = sys.argv[1]

# 1. init step
save_picture = False
show_results = True
save_path = __prefix_path__ + "resource/results/" + video_name
if not os.path.exists(save_path) and save_picture:
    os.makedirs(save_path)
net = init_net(gpus=0, rel_path=__prefix_path__)

# net = []
video_path = os.path.join(__prefix_path__, 'resource/video/' + video_name)
if not os.path.exists(video_path):
    print("No video")
    exit()
cap = cv2.VideoCapture(video_path)

first_frame = True
need_start = False
frame_state = {}
result_list = []
frame_id = 1
frame = []
while(cap.isOpened()):
    if not need_start:
        ret, frame = cap.read()

    frame_bbox = []
    if first_frame or need_start:
        if first_frame:
            first_frame = False
        else:
            need_start = False
        drawbox = drawBox('image%d' % frame_id, frame)
        while (True):
            temp = np.copy(frame)
            if drawbox.drawing and (drawbox.end_point != (-1, -1) and drawbox.start_point != (-1, -1)):
                cv2.rectangle(temp, drawbox.start_point, drawbox.end_point, (0, 255, 0), 1)
            cv2.imshow('image%d' % frame_id, temp)

            waitkey = cv2.waitKey(1) & 0xFF
            if waitkey == ord('q'):
                break
            elif waitkey == ord('n'):
                first_frame = True
                cv2.destroyAllWindows()
                break

            if not drawbox.drawing and drawbox.finish:
                print("please press 'R/r' to re-drawing, or others to continue !")
                waitkey_num = cv2.waitKey(0)
                print(waitkey_num)
                if waitkey_num == ord('R') or waitkey_num == ord('r'):
                    drawbox.finish = False
                    drawbox.draw_begin = False
                    frame = drawbox.backup_img
                    drawbox.img = drawbox.backup_img
                    drawbox.backup_img = np.copy(drawbox.img)
                    continue
                frame_bbox = drawbox.final_point
                cv2.destroyAllWindows()
                break
            elif not drawbox.finish and drawbox.draw_begin and not drawbox.drawing:
                print("Rect not right, Please re-do the box")
                drawbox.draw_begin = False

        if first_frame:
            result_list.append([-1,-1,-1,-1])
            print("The frame %d is skipped manually." % frame_id)
            continue

        result_list.append(frame_bbox)
        frame_state = init_track(net, frame, frame_bbox)
        if save_picture:
            if not save_pictures(frame, save_path, frame_id, frame_bbox):
                print("Save wrong")
                exit()
        frame_id += 1
        continue

    frame_state = SiamRPN_track(frame_state, frame)  # track
    if frame_state['score'] < 0.3:
        # need restart
        cv2.destroyAllWindows()
        need_start = True
    else:
        res = cxy_wh_2_rect(frame_state['target_pos'], frame_state['target_sz'])
        result_list.append(res)
        frame_id += 1
        if show_results and frame_id % 10 == 0:
            # cv2.destroyAllWindows()
            tmp_img = rect_box(frame, res, frame_state['score'], frame_id)
            width = tmp_img.shape[1]
            height = tmp_img.shape[0]
            cv2.namedWindow('results', 0)
            cv2.resizeWindow('results', width, height)
            cv2.imshow('results', tmp_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break





