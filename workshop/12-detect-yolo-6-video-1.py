from ultralytics import YOLO
import cv2


"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""


def detect_from_video(video_path,confiden=0.5,device='cpu'):
    ## load model
    cap = cv2.VideoCapture(video_path)
    model = YOLO('model/yolo11n.pt').to(device) ## Use cpu or gpu ('cuda')
    #predict with class
    model.predict(classes=[0,2])

    while cap.isOpened():
        success,frame = cap.read()
        if not success:
            print(" video error ")
            break 
      
        person_count = 0
        car_count = 0
      
        results = model(frame,iou=0.4,conf=confiden,verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() ## box
            scores = result.boxes.conf.cpu().numpy() ## conf
            classed = result.boxes.cls.cpu().numpy() ## classed
      
            
            for box,score,cls in zip(boxes,scores,classed):
                label = f"{model.names[int(cls)]}:{score:.2f}" ##ใช้สำหรับ puttext
                clsname = model.names[int(cls)] ##.ใช้สำหรับ if else by classed
                x1,y1,x2,y2 = map(int,box) ## ใช้สำหรับวาดสี่เหลียม
        
                
                ##### ไว้ตีกรอบหรือทำอะไรก็ได้ ####
                if clsname == "person":
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
                    cv2.putText(frame,f"{label}",(x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),1)
                if clsname == "car":
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),1)
                    cv2.putText(frame,f"{label}",(x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)
              
       
        cv2.imshow('YOLOV11 DETECT FROM VIDEO',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


detect_from_video('video/ronaldo2.mp4',0.3)    