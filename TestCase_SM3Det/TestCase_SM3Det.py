""" user import lib """
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
mmrotate_local_path = os.path.join(current_dir, 'mmrotate_local_bak')
if mmrotate_local_path not in sys.path:
    sys.path.insert(0, mmrotate_local_path)

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import mmrotate
""" user import lib done """

""" function user define """
def testSM3Det():
    configPath = "configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py"
    checkpointPath = "epoch_12.pth"
    
    if not os.path.exists(configPath):
        print(f"config file not found: {configPath}")
        return
    
    if not os.path.exists(checkpointPath):
        print(f"checkpoint file not found: {checkpointPath}")
        print("please download or train SM3Det checkpoint file")
        return
    try:
        model = init_detector(configPath, checkpointPath, device='cpu')
        print("model init success")
    except Exception as e:
        print(f"model init failed: {e}")
        return
    
    imgDir = r"C:\Users\user\Desktop\比賽\test\SM3Det\bdd100k\bdd100k\images\100k\test"
    
    if not os.path.exists(imgDir):
        print(f"image dir not found: {imgDir}")
        return
        
    imgList = sorted([f for f in os.listdir(imgDir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(imgList) == 0:
        print("no image file found")
        return
    
    print(f"find {len(imgList)} images")
    
    for i in range(min(3, len(imgList))):
        img_path = os.path.join(imgDir, imgList[i])
        print(f"process image: {imgList[i]}")
        
        try:
            result = inference_detector(model, img_path)    
            show_result_pyplot(
                model,
                img_path,
                result,
                palette='dota',
                score_thr=0.3,
                out_file=f"result_{i}.jpg"
            )
            
            print(f"image {imgList[i]} process done, result save as result_{i}.jpg")
            
        except Exception as e:
            print(f"error when process image {imgList[i]}: {e}")
            continue
        
""" function user define done"""

""" main """
if __name__ == "__main__":
    print("start test SM3Det")
    testSM3Det()
    print("test SM3Det done")
""" main done """
