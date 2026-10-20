from ultralytics import YOLO
import cv2
import numpy as np

# Load the optimized TensorRT Engine
# IMPORTANT: Use the .engine file, not the .pt file for deployment!
engine_path = "yolov8n.engine"
model = YOLO(engine_path, task='detect') 

def get_person_crop(denoised_frame: np.ndarray) -> np.ndarray:
    """
    Takes a BGR image (from NAFNet), detects the largest person, 
    and returns the cropped bounding box image array.
    """
    # Run inference. 
    # classes=[0] ensures it only looks for the 'person' class in COCO.
    # conf=0.5 sets a strict confidence threshold.
    results = model.predict(source=denoised_frame, classes=[0], conf=0.5, verbose=False)
    
    result = results[0]
    
    # Return None if no person is detected in this frame
    if len(result.boxes) == 0:
        return None
        
    # Extract bounding boxes in xyxy format (xmin, ymin, xmax, ymax)
    boxes = result.boxes.xyxy.cpu().numpy()
    
    # For a multi-angle re-ID pipeline, you typically want the largest subject 
    # (closest to camera) or you can iterate through all of them.
    largest_box = None
    max_area = 0
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)
            
    if largest_box:
        x1, y1, x2, y2 = largest_box
        
        # Add a 5% margin to the bounding box so GFPGAN has context for the face
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        
        img_h, img_w = denoised_frame.shape[:2]
        
        # Ensure coordinates don't go out of the frame bounds
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img_w, x2 + margin_x)
        y2 = min(img_h, y2 + margin_y)
        
        # Slice the numpy array to crop the person
        cropped_person = denoised_frame[y1:y2, x1:x2]
        
        return cropped_person
        
    return None

# --- Testing the Function ---
if __name__ == "__main__":
    # Load your NAFNet output image to test
    test_img = cv2.imread("test_denoised_input.png")
    
    if test_img is not None:
        person_crop = get_person_crop(test_img)
        
        if person_crop is not None:
            cv2.imshow("YOLO Cropped Person", person_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # This 'person_crop' array is what you will now feed into GFPGAN
        else:
            print("No person found in the frame.")
