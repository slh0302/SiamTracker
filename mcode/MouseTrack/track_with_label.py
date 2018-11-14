# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 10/15/18 2:45 PM

import os
import torch
import cv2
import sys
import numpy as np
from mcode.MouseTrack.utils.tools import drawBox, init_net, init_track, save_pictures, rect_box, save_results
from mcode.run_SiamRPN import SiamRPN_track
from mcode.utils import cxy_wh_2_rect

os.environ['CUDA_VISIBLE_DEVICES'] = '11,12'

__prefix_path__, _ = os.path.split(os.path.realpath(__file__))
video_name = sys.argv[1]

# 1. init step
save_picture = True
show_results = True
save_path = __prefix_path__ + "/resource/results/label/" + video_name.split('.')[0]
save_img_path = save_path + "/img"
if not os.path.exists(save_img_path) and save_picture:
    os.makedirs(save_img_path)
net = init_net(gpus=0, rel_path=__prefix_path__)

# net = []
video_path = os.path.join(__prefix_path__, 'resource/video/' + video_name)
if not os.path.exists(video_path):
    print("No video")
    exit()
cap = cv2.VideoCapture(video_path)

if '/' in video_name:
    video_name = video_name.split('/')[-1]

first_frame = True
need_start = False
frame_state = {}
result_list = []
frame_id = 1
frame = []

# re-boot function
save_file = video_name.split('.')[0] + ".txt"
tmp_file_path = os.path.join(save_path, save_file)
if os.path.exists(tmp_file_path) and os.path.getsize(tmp_file_path) > 5:
    with open(tmp_file_path, 'rb') as f:
        for line in f:
            sp = line[:-1].split()
            id = int(sp[0])
            x, y , w, h = int(sp[1]), int(sp[2]), int(sp[3]), int(sp[4])
            scores = float(sp[-1])
            result_list.append([id, x, y, w, h, scores])
        frame_id = result_list[-1][0] + 1
        print("Results Size: %d " % len(result_list))
        print("Restart at frame %d " % frame_id)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)


while(cap.isOpened()):
    if not need_start:
        ret, frame = cap.read()

    frame_bbox = []
    if first_frame or need_start:
        if first_frame:
            first_frame = False
        else:
            need_start = False
        print("Draw at frame id %d" % frame_id)
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
            result_list.append([frame_id, -1,-1,-1,-1,-1])
            print("The frame %d is skipped manually." % frame_id)
            frame_id += 1
            continue

        results = [frame_id] + frame_bbox + [1.0]
        result_list.append(results)
        frame_state = init_track(net, frame, frame_bbox)
        if save_picture:
            if not save_pictures(frame, save_img_path, frame_id, frame_bbox):
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
        results = [frame_id] + res.tolist()  + [frame_state['score']]
        result_list.append(results)
        if show_results and frame_id % 5 == 0:
            # cv2.destroyAllWindows()
            tmp_img = rect_box(frame, res, frame_state['score'], frame_id)
            width = tmp_img.shape[1]
            height = tmp_img.shape[0]
            cv2.namedWindow('results', 0)
            cv2.resizeWindow('results', width, height)
            cv2.imshow('results', tmp_img)
            wait_key = cv2.waitKey(1) & 0xFF
            if wait_key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif wait_key == ord('s'):
                before_frame_id = frame_id
                tmp_frame_id = before_frame_id
                while True:
                    wait_key_tmp = cv2.waitKey(1) & 0xFF
                    if wait_key_tmp == ord('p') and tmp_frame_id - 1 >= 0:
                        tmp_frame_id -= 1
                    elif wait_key_tmp == ord('n') and tmp_frame_id <= frame_id:
                        tmp_frame_id += 1
                    elif wait_key_tmp == ord('r'):
                        cv2.destroyAllWindows()
                        need_start = True
                        # frame_id = tmp_frame_id
                        while frame_id > 1 and result_list[frame_id - 1][0] > tmp_frame_id:
                            result_list.pop()
                            frame_id -= 1
                        result_list.pop()
                        print("return to frame id %d" % frame_id)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                        _, frame = cap.read()
                        break
                    elif wait_key_tmp == ord('q'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        cv2.destroyAllWindows()
                        break
                    if tmp_frame_id != before_frame_id:
                        print("Debug: frame id %d" % tmp_frame_id)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, tmp_frame_id - 1)
                        _, tmp_frame = cap.read()
                        tmp_img = rect_box(tmp_frame, result_list[tmp_frame_id - 1][1:-1], result_list[tmp_frame_id - 1][-1] , tmp_frame_id)
                        cv2.imshow('results', tmp_img)
                        before_frame_id = tmp_frame_id
        if not need_start:
            if save_picture:
                if not save_pictures(frame, save_img_path, frame_id, res):
                    print("Save wrong")
                    exit()
            if frame_id % 100 == 0:
                save_results(result_list, video_name, save_path)
            frame_id += 1

# print(result_list)
save_results(result_list, video_name, save_path)