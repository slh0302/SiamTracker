# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 10/18/18 3:20 PM

# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 2018/10/16 下午3:49


import cv2
import numpy as np

class drawBox:
    def __init__(self, window_name, img):
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.img  = img
        self.drawing = False
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_event)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if not (abs(x - self.start_point[0]) < 10 and abs(y - self.start_point[1]) < 10):
                cv2.rectangle(self.img, self.start_point, (x, y), (0, 255, 0), 1)
            self.end_point = self.start_point = (-1, -1)


drawing = False  # 是否开始画图
mode = True  # True：画矩形，False：画圆
final_rect = []
start = (-1, -1)
endp = (-1, -1)

def mouse_event(event, x, y, flags, param):
    global start, endp, drawing, mode

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            endp = (x,y)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if not (abs(x - start[0]) < 10 and abs(y - start[1]) < 10):
            cv2.rectangle(img, start, (x, y), (0, 255, 0), 1)
        endp = start = (-1, -1)


img = np.zeros((512, 512, 3), np.uint8)
db = drawBox('image', img)
while(True):
    temp = np.copy(img)
    if db.drawing and (db.end_point != (-1, -1) and db.start_point != (-1, -1)):
        cv2.rectangle(temp, db.start_point, db.end_point, (0,255,0), 1)
    cv2.imshow('image', temp)

    # 按下m切换模式
    if cv2.waitKey(1) == ord('m'):
        mode = not mode
    elif cv2.waitKey(1) == 27:
        break
