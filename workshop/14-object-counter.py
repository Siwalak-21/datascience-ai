import cv2 # นำเข้าไลบรารี OpenCV สำหรับงานประมวลผลภาพและวิดีโอ

from ultralytics import solutions # นำเข้าโมดูล 'solutions' จากไลบรารี ultralytics ซึ่งมีเครื่องมือสำหรับงาน AI เช่น Object Counter

model = 'model/yolo11n.pt' # กำหนดพาธไปยังไฟล์โมเดล YOLOv11 ที่จะใช้ในการตรวจจับวัตถุ (ในที่นี้คือ 'yolo11n.pt' ซึ่งน่าจะเป็นโมเดล YOLOv8 nano)

def count_objects_in_region(video_path, output_video_path, model_path): # ฟังก์ชันสำหรับนับวัตถุในพื้นที่ที่กำหนดภายในวิดีโอ โดยรับ 3 พารามิเตอร์: พาธวิดีโออินพุต, พาธวิดีโอเอาต์พุต, และพาธโมเดล
    """Count objects in a specific region within a video.""" # Docstring อธิบายว่าฟังก์ชันนี้ทำอะไร

    cap = cv2.VideoCapture(video_path) # สร้างอ็อบเจกต์ VideoCapture เพื่อเปิดไฟล์วิดีโอจาก 'video_path'
    assert cap.isOpened(), "Error reading video file" # ตรวจสอบว่าวิดีโอเปิดได้สำเร็จหรือไม่ ถ้าไม่สำเร็จจะแสดงข้อความ Error
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) # ดึงความกว้าง (w), ความสูง (h), และอัตราเฟรม (fps) ของวิดีโอ
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) # สร้างอ็อบเจกต์ VideoWriter เพื่อเขียนเฟรมวิดีโอที่ประมวลผลแล้วลงใน 'output_video_path' โดยใช้ codec "mp4v"

    # กำหนดจุดของพื้นที่ที่ต้องการนับวัตถุ (เป็นพิกัด x, y ของมุมทั้งสี่ของรูปสี่เหลี่ยม)
    region_points = [(20, 400), (1280, 400), (1280, 360), (20, 360)]
    # สร้างอ็อบเจกต์ ObjectCounter โดยกำหนด:
    # show=True เพื่อแสดงผลลัพธ์บนหน้าต่าง (ถ้ามี)
    # region=region_points เพื่อกำหนดพื้นที่ที่ต้องการนับวัตถุ
    # model=model_path เพื่อระบุโมเดล YOLO ที่ใช้ในการตรวจจับ
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened(): # วนลูปตราบเท่าที่วิดีโอยังคงเปิดอยู่
        success, im0 = cap.read() # อ่านเฟรมถัดไปจากวิดีโอ: 'success' จะเป็น True ถ้าอ่านสำเร็จ, 'im0' คือเฟรมภาพ
        if not success: # ถ้าอ่านเฟรมไม่สำเร็จ (เช่น วิดีโอจบแล้วหรือมีปัญหา)
            print("Video frame is empty or processing is complete.") # แสดงข้อความแจ้งว่าวิดีโอจบแล้ว
            break # ออกจากลูป
        results = counter(im0) # ส่งเฟรม 'im0' ไปให้ ObjectCounter เพื่อประมวลผล (ตรวจจับและนับวัตถุ)
        video_writer.write(results.plot_im) # เขียนเฟรมที่มีผลลัพธ์การตรวจจับและนับวัตถุ (พร้อมการแสดงผล) ลงในไฟล์วิดีโอเอาต์พุต

        if cv2.waitKey(10) & 0xFF == ord('q'): # รอรับการกดปุ่มเป็นเวลา 1 มิลลิวินาที:
            # ถ้ามีการกดปุ่ม 'q' (แปลงเป็น ASCII และเปรียบเทียบ)
            break # ออกจากลูปเพื่อหยุดการประมวลผล
    cap.release() # ปล่อยทรัพยากรของอ็อบเจกต์ VideoCapture (ปิดไฟล์วิดีโออินพุต)
    video_writer.release() # ปล่อยทรัพยากรของอ็อบเจกต์ VideoWriter (ปิดไฟล์วิดีโอเอาต์พุต)ๆ
    cv2.destroyAllWindows() # ปิดหน้าต่าง OpenCV ที่เปิดอยู่ทั้งหมด

# เรียกใช้ฟังก์ชัน count_objects_in_region ด้วยพาธวิดีโออินพุต "video/car.mp4", พาธวิดีโอเอาต์พุต "output_video.avi" และโมเดลที่กำหนดไว้
count_objects_in_region("video/car.mp4", "output_video.avi", model)