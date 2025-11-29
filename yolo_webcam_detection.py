import cv2
from ultralytics import YOLO

# --- அமைப்புகளை இங்கே மாற்றவும் ---
# உங்கள் DroidCam அல்லது பிற வெப்கேம் இன்டெக்ஸ் எண்ணைக் கொடுக்கவும்.
# பொதுவாக 0 (லேப்டாப் கேம்), 1 (DroidCam) என இருக்கும்.
WEBCAM_INDEX = 1 

# பயன்படுத்த வேண்டிய YOLO மாடலின் பெயரை இங்கு குறிப்பிடவும்.
# (எ.கா: 'yolov8n.pt' - நானோ, 'yolov8s.pt' - சிறியது)
MODEL_NAME = 'yolov8n.pt' 
# -----------------------------------

def yolo_webcam_detection(webcam_index, model_name):
    """
    வெப்கேம் உள்ளீட்டில் YOLOv8 ஆப்ஜெக்ட் கண்டறிதலைச் செய்கிறது.
    """
    
    # 1. YOLOv8 மாடலை ஏற்றுதல் (Load the YOLOv8 model)
    try:
        model = YOLO(model_name)
        print(f"YOLO மாடல் '{model_name}' வெற்றிகரமாக ஏற்றப்பட்டது.")
    except Exception as e:
        print(f"YOLO மாடலை ஏற்றுவதில் பிழை: {e}")
        return

    # 2. வெப்கேம் ஸ்ட்ரீமைத் தொடங்குதல் (Start Webcam Stream)
    cap = cv2.VideoCapture(webcam_index)
    
    # வெப்கேம் இணைக்கப்பட்டுள்ளதா எனச் சரிபார்க்கவும்
    if not cap.isOpened():
        print(f"பிழை: வெப்கேம் இன்டெக்ஸ் {webcam_index} திறக்க முடியவில்லை. சரியான எண்ணைச் சரிபார்க்கவும்.")
        return

    print("வெப்கேம் ஸ்ட்ரீம் தொடங்கப்பட்டது. முடிவடைய 'q' அழுத்தவும்.")
    
    # 3. ஸ்ட்ரீம் செயலாக்கம் (Process the Stream)
    while True:
        # ஃப்ரேமைப் படித்தல் (Read frame from the camera)
        ret, frame = cap.read()
        
        # ஃப்ரேம் படிக்க முடியவில்லை என்றால், லூப்பை விட்டு வெளியேறவும்
        if not ret:
            print("ஃப்ரேம் ஸ்ட்ரீமில் இருந்து படிக்க முடியவில்லை. வெளியேறுகிறது.")
            break
        
        # 4. YOLO இன்ஃபரன்ஸ் (Perform YOLO Inference)
        # YOLO மாடலுக்கு ஃப்ரேமை உள்ளீடாகக் கொடுத்து முடிவுகளைப் பெறுதல்
        results = model(frame, verbose=False) # verbose=False அமைதியைச் சத்தமில்லாமல் இயக்க உதவும்
        
        # 5. முடிவுகளை ஃப்ரேமில் வரைதல் (Plot results on the frame)
        # ultralytics தானாகவே பெட்டிகள் (boxes) மற்றும் லேபிள்களை (labels) வரையும்
        annotated_frame = results[0].plot()
        
        # 6. முடிவுகளைக் காண்பித்தல் (Display the output)
        cv2.imshow("YOLO Live Detection - (Q to Quit)", annotated_frame)
        
        # 'q' பொத்தானை அழுத்தினால் லூப்பை விட்டு வெளியேறவும்
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. cleanup செய்தல் (Release resources)
    cap.release()
    cv2.destroyAllWindows()
    print("கண்டறிதல் நிறுத்தப்பட்டு, அனைத்து ஜன்னல்களும் மூடப்பட்டன.")

# கோடை இயக்குதல் (Run the script)
if __name__ == "__main__":
    yolo_webcam_detection(WEBCAM_INDEX, MODEL_NAME)