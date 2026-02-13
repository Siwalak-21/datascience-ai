import cv2
## Face Detection
face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
## อ่านภาพ
img = cv2.imread('photos/manu.jpg')
## แปลงภาพเป็นขาวดำ
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## ตรวจจับใบหน้า
faces = face.detectMultiScale(gray,1.1,4)

counter = 1
## วาดสี่เหลี่ยมรอบใบหน้าและ save
for(x,y,w,h) in faces:
    ## save ใบหน้า
    face_img = img[y:y+h,x:x+w]
    ## ชื่อไฟล์
    filename = f"faces_save/face{counter}.jpg"
    ## บันทึกภาพ
    cv2.imwrite(filename,face_img)
    ## นับจำนวนใบหน้า
    counter+=1
    ## แสดงชื่อไฟล์ที่บันทึก
    print(f"saved: {filename}")
    ## วาดสี่เหลี่ยมรอบใบหน้า
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

## แสดงภาพ
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()   