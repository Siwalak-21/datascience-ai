import cv2
from ultralytics import YOLO

# Load your custom trained model
model = YOLO("model/best.pt")  # เปลี่ยนเป็น best.pt ที่คุณเทรนมา

# Initialize camera (0 for default camera, 1 for external camera)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'q' to quit")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Perform detection with your chosen confidence threshold
    results = model(frame, conf=0.9)
    
    # Draw detection results on frame
    annotated_frame = results[0].plot()
    
    # ✨ --- ส่วนที่เพิ่มเข้ามา --- ✨
    
    # 1. นับจำนวนวัตถุที่ตรวจจับได้ในเฟรมนี้
    object_count = len(results[0].boxes)
    
    # 2. เตรียมข้อความที่จะแสดง
    count_text = f"Count: {object_count}"
    
    # 3. วาดข้อความลงบนเฟรม (annotated_frame)
    cv2.putText(
        annotated_frame,
        count_text,
        (50, 50),  # ตำแหน่ง (x, y) ที่จะวาดข้อความ
        cv2.FONT_HERSHEY_SIMPLEX,  # รูปแบบตัวอักษร
        1.5,  # ขนาดตัวอักษร
        (0, 255, 0),  # สีของข้อความ (BGR) - ในที่นี้คือสีเขียว
        3,  # ความหนาของเส้น
        cv2.LINE_AA  # ทำให้ขอบตัวอักษรเรียบเนียน
    )
    
    # ✨ --- สิ้นสุดส่วนที่เพิ่มเข้ามา --- ✨

    # Display the frame with the count
    cv2.imshow('Meter Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Camera released successfully")