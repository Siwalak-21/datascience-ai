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
    
    # Perform detection with lower confidence threshold
    results = model(frame, conf=0.9)  # ลดค่า confidence ลง
    
    # Print detection info for debugging
    if len(results[0].boxes) > 0:
        print(f"Detected {len(results[0].boxes)} objects")
        for box in results[0].boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            print(f"Class: {class_id}, Confidence: {confidence:.3f}")
    else:
        print("No detections found")
    
    # Draw detection results on frame
    annotated_frame = results[0].plot()
    
    # Display the frame
    cv2.imshow('Meter Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Camera released successfully")