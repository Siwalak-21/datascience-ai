import cv2
## ที่อยู่ของวีดีโอ
video_path ="video/ronaldo2.mp4"

## อ่านวีดีโอ
cap = cv2.VideoCapture(video_path)

while True:
    success,frame = cap.read()
    ## ถ้าอ่านวีดีโอไม่สำเร็จ
    if not success:
        print(" read video error")
        break
    
    ## แสดงวีดีโอ
    cv2.imshow("video",frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
## ปิดวีดีโอ    
cap.release()
cv2.destroyAllWindows    