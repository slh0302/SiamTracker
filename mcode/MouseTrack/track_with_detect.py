# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 10/18/18 4:50 PM

import os
import cv2
import sys
import torch
import numpy as np
from mcode.utils import cxy_wh_2_rect
from mcode.run_SiamRPN import SiamRPN_track
from mcode.MouseTrack.utils.detector import faster_rcnn, check_confidence
from mcode.MouseTrack.utils.tools import drawBox, init_net, init_track, judge_detect_frame, judge_track_frame, rect_box


""" 0. some variable """
os.environ['CUDA_VISIBLE_DEVICES'] = '14,15'

if len(os.environ['CUDA_VISIBLE_DEVICES']) >= 2:
    detect_gpu_id = 1
else:
    detect_gpu_id = 0

__prefix_path__, _ = os.path.split(os.path.realpath(__file__))
video_name = sys.argv[1]
direction = sys.argv[2]
if len(sys.argv) > 3:
    foot_type = sys.argv[3]
else:
    if direction == "right":
        foot_type = 'front_foot_right'
    else:
        foot_type = 'front_foot_left'

save_picture = False
show_results = False
show_track_results = True
save_path = __prefix_path__ + "resource/results/" + video_name
detect_path = os.path.join(__prefix_path__, 'utils/model/', direction, 'faster_rcnn.pth')
video_path = os.path.join(__prefix_path__, 'resource/video/' + video_name)


""" 1. init track network step """
if not os.path.exists(save_path) and save_picture:
    os.makedirs(save_path)
track_net = init_net(gpus=0, rel_path=__prefix_path__)


""" 2. init detection network step """
detect_net = faster_rcnn(1, False, base_net=101, load_name=detect_path, gpu_id=detect_gpu_id)
print("Done init.")


""" 3. Load Video """
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
track_res = []
before_frame = []
track_frame_res = {}
before_frame_box = []
while(cap.isOpened()):
    if not need_start:
        ret, frame = cap.read()

    frame_bbox = []
    if first_frame or need_start:
        detect_before = False
        if first_frame:
            first_frame = False
        else:
            need_start = False
            detect_before = True

        """ frame detection  """

        res = detect_net.detect_img(frame, gpus=1)
        res = check_confidence(res)

        if detect_before:
            before_res = detect_net.detect_img(before_frame)
            before_res = check_confidence(before_res)
            track_frame_res = judge_track_frame(before_frame_box, before_res, foot_type)

        if foot_type in res.keys():
            frame_bbox = res[foot_type]
            frame_bbox = judge_detect_frame(track_frame_res, frame_bbox)
        else:
            frame_bbox = [1, 1, 1, 1, 0]
            print("Wrong Detect.")

        """ show results and modify """
        while show_results :
            show_res = rect_box(frame, frame_bbox[:-1], frame_bbox[-1], frame_id)
            width = show_res.shape[1]
            height = show_res.shape[0]
            cv2.namedWindow('results', 0)
            cv2.resizeWindow('results', width, height)
            cv2.imshow('results-%d' % frame_id, show_res)
            waitkey = cv2.waitKey(1) & 0xFF
            if waitkey == ord('q'):
                """ when Detection result is all right. """
                cv2.destroyAllWindows()
                break
            elif waitkey == ord('R') or waitkey == ord('r'):
                """ Modify """
                cv2.destroyAllWindows()
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
                break

            elif waitkey == ord('n'):
                """ when foot part is not show in the picture. """
                first_frame = True
                cv2.destroyAllWindows()
                break

        if first_frame:
            result_list.append([-1, -1, -1, -1])
            print("The frame %d is skipped manually." % frame_id)
            continue

        if frame_bbox[-1] == 0:
            first_frame = True
            frame_bbox = [-1,-1,-1,-1]
            result_list.append(frame_bbox[:4])
            continue


        track_res = frame_bbox
        before_frame = frame
        result_list.append(frame_bbox[:4])
        frame_state = init_track(track_net, frame, frame_bbox)
        frame_id += 1
        continue

    """ Tracking """
    before_frame_box = track_res
    frame_state = SiamRPN_track(frame_state, frame)  # track
    if frame_state['score'] < 0.3:
        # need restart
        cv2.destroyAllWindows()
        need_start = True
    else:
        track_res = cxy_wh_2_rect(frame_state['target_pos'], frame_state['target_sz'])
        # detect_res, sit = judge_detect_res(detect_net, res_dict['res'][-1], sequences_list[index:], foot_type)
        result_list.append(track_res)
        before_frame = frame
        frame_id += 1
        if show_track_results and frame_id % 10 == 0:
            # cv2.destroyAllWindows()
            tmp_img = rect_box(frame, track_res, frame_state['score'], frame_id)
            width = tmp_img.shape[1]
            height = tmp_img.shape[0]
            cv2.namedWindow('results', 0)
            cv2.resizeWindow('results', width, height)
            cv2.imshow('results', tmp_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break



