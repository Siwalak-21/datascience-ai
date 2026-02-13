import cv2
from ultralytics import YOLO

# --- การตั้งค่า ---
MODEL_PATH = "model/best.pt"
CONFIDENCE_THRESHOLD = 0.8 # อาจจะต้องปรับค่านี้ตามความเหมาะสม

# --- เริ่มต้นโปรแกรม ---
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# ✨ 1. สร้างตัวแปรสำหรับ State Machine
# สถานะเริ่มต้นคือ "กำลังรอวัตถุ"
current_state = "WAITING" 
# ยอดรวมสะสมของ "เหตุการณ์" ที่นับได้
total_event_count = 0

print("Event Counting a.k.a State Machine method.")
print("Waiting for object...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ใช้ model() ธรรมดาก็เพียงพอแล้ว ไม่จำเป็นต้องใช้ .track()
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    annotated_frame = results[0].plot()
    
    # ตรวจสอบว่าในเฟรมปัจจุบันมีวัตถุหรือไม่
    is_object_detected = len(results[0].boxes) > 0
    
    # ✨ 2. Logic ของ State Machine
    # กรณีที่ 1: สถานะคือ "กำลังรอ" และ "เจอวัตถุ"
    if current_state == "WAITING" and is_object_detected:
        print(f"EVENT DETECTED! -> Changing state to OBJECT_PRESENT")
        # บวกยอดรวมเพิ่ม 1
        total_event_count += 1
        # เปลี่ยนสถานะเป็น "วัตถุปรากฏแล้ว" เพื่อไม่ให้นับซ้ำ
        current_state = "OBJECT_PRESENT"
        
    # กรณีที่ 2: สถานะคือ "วัตถุปรากฏแล้ว" และ "ไม่เจอวัตถุ"
    elif current_state == "OBJECT_PRESENT" and not is_object_detected:
        print(f"OBJECT REMOVED. -> Changing state back to WAITING")
        # เปลี่ยนสถานะกลับไปเป็น "กำลังรอ" เพื่อให้พร้อมนับชิ้นต่อไป
        current_state = "WAITING"

    # ✨ 3. แสดงผลยอดรวมของ "เหตุการณ์"
    cv2.putText(annotated_frame, f"Total Count: {total_event_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    # แสดงสถานะปัจจุบันบนจอ (สำหรับ Debug)
    cv2.putText(annotated_frame, f"State: {current_state}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Event Counter', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()