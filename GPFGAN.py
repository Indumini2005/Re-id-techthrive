import cv2
import numpy as np
from gfpgan import GFPGANer

# Initialize the GFPGANer object globally so the model loads into the GPU only once
# We disable the bg_upsampler (Real-ESRGAN) because we only care about the face crop
face_enhancer = GFPGANer(
    model_path='experiments/pretrained_models/GFPGANv1.3.pth', 
    upscale=2,               # Upscales the face crop by 2x for ArcFace
    arch='clean', 
    channel_multiplier=2,
    bg_upsampler=None        
)

def restore_yolo_crop(yolo_cropped_img: np.ndarray) -> np.ndarray:
    """
    Takes the BGR NumPy array from YOLOv8, aligns it, and hallucinates 
    missing high-resolution facial features.
    """
    # The enhance() method handles the face alignment and StyleGAN2 synthesis
    # has_aligned=False tells GFPGAN to align the face itself using facexlib
    _, restored_faces, _ = face_enhancer.enhance(
        yolo_cropped_img, 
        has_aligned=False, 
        only_center_face=True, 
        paste_back=False
    )
    
    # If facexlib successfully detected and enhanced a face within the YOLO crop
    if restored_faces:
        return restored_faces[0] # Return the high-res BGR array
        
    return None

# --- Testing the Integration ---
if __name__ == "__main__":
    # Assume 'person_crop' is the NumPy array output from your YOLO script
    person_crop = cv2.imread("test_yolo_output.png")
    
    if person_crop is not None:
        high_res_face = restore_yolo_crop(person_crop)
        
        if high_res_face is not None:
            cv2.imshow("GFPGAN Restored Face", high_res_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # -> Next step: Send 'high_res_face' to ArcFace!
        else:
            print("GFPGAN could not find a distinct face inside the YOLO crop.")
