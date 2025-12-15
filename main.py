"""
Desktop Object Detection using OpenCV and MediaPipe
Real-time object detection from webcam feed with bounding box visualization
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    """Main function to run desktop object detection"""
    
    # Initialize MediaPipe Object Detector
    base_options = python.BaseOptions(
        model_asset_path='efficientdet_lite0.tflite'
    )
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        max_results=5,
        score_threshold=0.5
    )
    detector = vision.ObjectDetector.create_from_options(options)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting object detection... Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect objects
        detection_result = detector.detect_for_video(mp_image, frame_count)
        frame_count += 1
        
        # Draw detection results
        for detection in detection_result.detections:
            # Get bounding box
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get category and score
            category = detection.categories[0]
            category_name = category.category_name
            score = round(category.score, 2)
            
            # Draw label
            label = f"{category_name}: {score}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Object Detection - Press q to quit', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
