import cv2
from ultralytics import YOLO

# --- ส่วนสำคัญ ---
# โหลดโมเดล yolov11n-pose ซึ่งเป็นโมเดลมาตรฐาน
# หากคุณยังไม่มีไฟล์นี้ในเครื่อง ไลบรารี ultralytics จะดาวน์โหลดให้โดยอัตโนมัติ!
model = YOLO('model/yolo11n-pose.pt')
# -----------------

# เปิดการใช้งานกล้องเว็บแคม (เลข 0 คือกล้องตัวหลัก)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video/ronaldo2.mp4")

# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้ (Cannot open camera)")
    exit()

print("กำลังเปิดกล้อง... กด 'q' ที่หน้าต่างวิดีโอเพื่อปิดโปรแกรม")

while True:
    # อ่านเฟรมภาพจากกล้องทีละเฟรม
    success, frame = cap.read()
    if not success:
        print("ไม่สามารถอ่านเฟรมจากกล้องได้ (Failed to grab frame)")
        break

    # พลิกภาพในแนวนอน (ซ้าย-ขวา) เพื่อให้เหมือนมองกระจก
    frame = cv2.flip(frame, 1)

    # สั่งให้โมเดลทำการตรวจจับท่าทางในเฟรมปัจจุบัน
    # verbose=False เพื่อไม่ให้แสดง log ที่ไม่จำเป็นใน console
    results = model(frame, stream=True, verbose=False)

    # วนลูปผลลัพธ์ที่ได้ (แม้จะมีแค่เฟรมเดียว แต่ stream=True จะคืนค่าเป็น generator)
    for r in results:
        # ใช้ฟังก์ชัน plot() เพื่อวาด keypoints และเส้นเชื่อมต่อบนภาพโดยอัตโนมัติ
        annotated_frame = r.plot()

        # แสดงภาพที่วาดผลลัพธ์แล้วในหน้าต่างชื่อ 'YOLOv11 Body Pose'
        cv2.imshow('YOLOv11 Body Pose', annotated_frame)

    # รอรับการกดปุ่ม 'q' เพื่อออกจากลูปและปิดโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# คืนทรัพยากรกล้องและปิดหน้าต่างทั้งหมด
print("กำลังปิดโปรแกรม...")
cap.release()
cv2.destroyAllWindows()