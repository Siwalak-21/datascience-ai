import cv2
## ที่อยู่ของวีดีโอ
video_path ="video/ronaldo.mp4"


## อ่านวีดีโอ
cap = cv2.VideoCapture(video_path)


## Face Detection
face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

while True:
    success,frame = cap.read()
    ## ถ้าอ่านวีดีโอไม่สำเร็จ
    if not success:
        print(" read video error")
        break
    
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    faces = face.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),3)
    
    ## แสดงวีดีโอ
    cv2.imshow("Video Face Detection",frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
## ปิดวีดีโอ    
cap.release()
cv2.destroyAllWindows    