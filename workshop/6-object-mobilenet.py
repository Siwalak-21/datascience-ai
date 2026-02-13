import numpy as np 
import cv2

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
    "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
    "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
    "SOFA", "TRAIN", "TVMONITOR"]

COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD.prototxt","MobileNetSSD/MobileNetSSD.caffemodel")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    
    if ret: 
        (h,w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5) # Converts the 'frame' image into a blob, a format that the DNN model can understand.
        net.setInput(blob) # Feeds the converted blob to the DNN model ('net' is the variable storing the model).
        detections = net.forward() # Instructs the model to make predictions. The results are stored in the 'detections' variable.

        for i in np.arange(0, detections.shape[2]): 
            percent = detections[0,0,i,2] 
            
            if percent > 0.7:
                class_index = int(detections[0,0,i,1]) 
                box = detections[0,0,i,3:7]*np.array([w,h,w,h]) 
                (startX, startY, endX, endY) = box.astype("int") 

                
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                cv2.rectangle(frame, (startX-1, startY-40), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                y = startY - 15 if startY-15>15 else startY+15 
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)

                print(label)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
