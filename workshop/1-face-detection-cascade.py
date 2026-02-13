import cv2
## Face Detection
face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
## อ่านภาพ
img = cv2.imread('photos/manu.jpg')
## แปลงภาพเป็นขาวดำ
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## ตรวจจับใบหน้า
faces = face.detectMultiScale(gray,1.1,4)
## วาดสี่เหลี่ยมรอบใบหน้า
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

## แสดงภาพ
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()   