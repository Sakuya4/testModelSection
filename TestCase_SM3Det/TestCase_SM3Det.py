""" user import lib """
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
""" user import lib done """

""" function user define """
def testSM3Det():
    configPath = "SM3Det/configs/SM3Det/SM3Det_convnext_t.py"
    checkpointPath = "SM3Det.pth"
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


if __name__ == "__main__":
    print(" --- start test SM3Det")
    testSM3Det()
    print(" --- test SM3Det done")

