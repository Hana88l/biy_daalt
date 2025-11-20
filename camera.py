# from ultralytics import YOLO
# import cv2

# model = YOLO('yolov8n.pt')  


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Вебкам уншигдсангүй!")
#         break

#     results = model(frame)

#     annotated_frame = results[0].plot()  

#     cv2.imshow("YOLOv8 Object Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import time

def main_optimized():
    model = YOLO('yolov8n.pt')  
    
    # Use smaller model for better performance
    # model = YOLO('yolov8n.pt')  # nano (fastest)
    # model = YOLO('yolov8s.pt')  # small
    # model = YOLO('yolov8m.pt')  # medium
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("Optimized detection running. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
        
        # Perform inference with confidence threshold
        results = model(frame, conf=0.5)  # Only show detections with >50% confidence
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Display performance info
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {len(results[0])}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_optimized()