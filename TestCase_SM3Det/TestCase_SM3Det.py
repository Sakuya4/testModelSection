""" user import lib """
import cv2
import numpy as np
import torch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
mmrotate_local_path = os.path.join(current_dir, 'mmrotate_local_bak')
if mmrotate_local_path not in sys.path:
    sys.path.insert(0, mmrotate_local_path)

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmrotate
""" user import lib done """

""" function user define """
def testSM3Det():
    configPath = "configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py"
    checkpointPath = "epoch_12.pth"
    
    if not os.path.exists(configPath):
        print(f"config file not found: {configPath}")
        return None
    
    if not os.path.exists(checkpointPath):
        print(f"checkpoint file not found: {checkpointPath}")
        print("please download or train SM3Det checkpoint file")
        return None
    try:
        model = init_detector(configPath, checkpointPath, device='cpu')
        print("model init success")
    except Exception as e:
        print(f"model init failed: {e}")
        return None
    
    imgDir = r"../bdd100k/bdd100k/images/100k/test"
    
    if not os.path.exists(imgDir):
        print(f"image dir not found: {imgDir}")
        return None
        
    imgList = sorted([f for f in os.listdir(imgDir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(imgList) == 0:
        print("no image file found")
        return None
    
    print(f"find {len(imgList)} images")
    
    for i in range(min(5, len(imgList))):
        imgPath = os.path.join(imgDir, imgList[i])
        print(f"processing {imgPath}")
        processed_img_path = process_image(imgPath)
        result = inference_detector(model, processed_img_path)
        out_file = f"result_{i}.jpg"
        show_result_pyplot(
            model, processed_img_path, result,
            score_thr=0.1,
            out_file=out_file
        )
        print(f"image {imgList[i]} process done, result save as {out_file}")
    
    return model

def process_image(imgPath):
    img = cv2.imread(imgPath)
    if img is None:
        print(f"can't read image: {imgPath}")
        return imgPath
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    base_name = os.path.basename(imgPath)
    processed_dir = "processed"
    os.makedirs(processed_dir, exist_ok=True)
    processed_img_path = os.path.join(processed_dir, f"processed_{base_name}")
    cv2.imwrite(processed_img_path, img)
    
    print(f"image process done: {processed_img_path}")
    return processed_img_path

def main():
    model = testSM3Det()
    if model is None:
        print("model init failed, can't detect")
        return

""" function user define done"""

""" main """
if __name__ == "__main__":
    print("start test SM3Det model")
    main()
    print("test SM3Det model done")
""" main done """
