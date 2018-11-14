# Tracking Toolbox User Guide
## 0. Installation Guide
**Basic Requirements:**
```
1. Linux system with GUI.(not test on windows, but should works.)
2. Python3.x
3. Pytorch-4.0
4. GPU required --- CUDA is installed in /usr/local/cuda
```

**Linux :**
```
1. git clone https://github.com/slh0302/SiamTracker.git (Use the provided packages better).
2. cd SiamTracker
3. pip install -r requirements.txt
4. cd mcode/MouseTrack/lib
5. sh ./make.sh
```

**Read every tips carefully!**

## 1. Tracking methods

#### 0. Preparations

0. Put this project path to PYTHONPATH: `export PYTHONPATH=/path/SiamTracker`
1. Make resource directory:
  ```
  mkdir -p mcode/MouseTrack/resource/video \
           mcode/MouseTrack/utils/model/siam \
           mcode/MouseTrack/utils/model/right \
           mcode/MouseTrack/utils/model/left
  ```
2. Download tracking model and detection model, save to specified directory.
  Tracking model [[SiamRPNBIG.model](https://drive.google.com/open?id=1cXySnXG6L6hjmc4Lp94Yw3CnS6kZqM2-)]: save to `mcode/MouseTrack/utils/model/siam`, Detection model[[left side(**not provided yet**)]|[right side(faster_rcnn.pth)](https://drive.google.com/open?id=1S0CobZ2Dd8umcRs0nDKvRDAl-KVli1Yl)]: save to `mcode/MouseTrack/utils/model/right(or left)`. Pretrained-resnet model for training faster_rcnn detector: [resnet101_caffe.pth](https://drive.google.com/open?id=1VGeOjBaiPhZSDPXdf75WEd8RDKHbJlfS), save it to `mcode/MouseTrack/utils/model/detector`. Using example videos: [example.avi](https://drive.google.com/open?id=1FYpHUJb719KUxEcDe2Y72LpuoCja_h3Q), save this video to `mcode/MouseTrack/resource/video`.

3. Put videos to this path: `mcode/MouseTrack/resource/video` (The video's name should be the param of run scripts expressed as `video_name`)

#### 1. Some demo for User
**Tracking Type 1: With manually labeling**
1. This tracking method is modifed from DaSiamRPN.
```
cd mcode/MouseTrack
python ./track_with_label.py video_name
```

2. How to use:

  1). Using mouse to draw the part you want to track. Press 'r/R' to re-draw box, or any other keys to continue(not 'esc' or 'q').

  2). Run scripts when tracking failed, and you can draw the correct box again, or press 'n' to skip the frame when no object is found in this frame.

  3). This scripts will save the results' file in the following manner(splits by space):
    ```
    frame_id x y w h confidence
    ```
    confidences : 0~1 for confidence of this target, -1 means skipped or failed.

  **4). Tips**: while Tracking scrips running, you can still press 's' to pause the program, and then use 'p' for previous frame or 'n' for next frame, also you can press 'q' to continue runing traking from 'last' frame or press 'r' to re-start manually from the frame you think is not correctly track.


**Tracking Type 2: With auto-detect:**

1. We provide the Faster-RCNN detector(trained in mouse videos), you can use any other detector to replace this.(Modify the code yourself.)
```
cd mcode/MouseTrack
python ./track_with_detect.py video_name [left|right]
```
(**Tips**: Second param is 'left' or 'right')

2. The running results are the same as previous method.
3. Tips: You can pause the program by pressing 's' and 'q' for continue.

**Extra Function: Train Detector**

1. We provide the Faster-RCNN detector's training code. You can either use this detector or use yours.
2. This code is Slightly modifed from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch.git).
3. Need Pascal VOC2007 datasets' format: [example dataset(left-side)](https://drive.google.com/open?id=1CB8OGS36aKJyC8iEzOJUSq5epMxK8Wfr), save it to `mcode/MouseTrack/resource/datasets/left`. Training from scratch:
```
**With Multi-GPUs**
python ./train_detector.py --nw 10 --bs 16 --net res50 --epochs 20 --side left --lr 0.001 --cuda --mGPUs
**with single-Gpu**
python ./train_detector.py --nw 10 --bs 4 --net res50 --epochs 20 --side left --lr 0.001 --cuda
```
4. Fine-tuning:
```
python ./train_detector.py  --side left --nw 10 --bs 16 --net res50 --epochs 30 \
                              --lr 0.001 --cuda --mGPUs --start_epoch 20 --r True \
                              --checksession 1 --checkpoint 133 --checkepoch 25
```
5. **Tips-1:** There're more args you can use for training, please see function `def parse_args():` in train_detector.py for detail.

6. **Tips-2:** The default model path is `/home/slh/torch/SiamTracker/mcode/MouseTrack/utils/model/left`, the last directory is depend on the parm `--side left` or `--side right`. You're supposed to make these directories first.
