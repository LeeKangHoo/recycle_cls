from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = 'best_new_large.pt'

try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: '{MODEL_PATH}'")
    print(f"Categories: {model.names}")
except Exception as e:
    print(f"Error: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("\n" + "=" * 60)
print("Smart Recycling Station - UWCD Model")
print("=" * 60)
print("Press 'q' to quit\n")

CONFIDENCE_THRESHOLD = 0.50
PAPER_PENALTY = 0.90

categories = {
    'plastic': {
        'name': 'PLASTIC/VINYL',
        'korean': '플라스틱/비닐',
        'color': (51, 153, 255),
        'bin': '[PLASTIC BIN]',
        'instruction': 'Clean and dry before disposal'
    },
    'glass': {
        'name': 'GLASS',
        'korean': '유리',
        'color': (0, 255, 0),
        'bin': '[GLASS BIN]',
        'instruction': 'Remove caps and labels'
    },
    'metal': {
        'name': 'METAL',
        'korean': '금속(캔)',
        'color': (203, 192, 255),
        'bin': '[METAL BIN]',
        'instruction': 'Crush cans to save space'
    },
    'paper': {
        'name': 'PAPER',
        'korean': '종이',
        'color': (255, 200, 100),
        'bin': '[PAPER BIN]',
        'instruction': 'Keep dry and flat'
    },
    'trash': {
        'name': 'GENERAL WASTE',
        'korean': '일반 쓰레기',
        'color': (128, 128, 128),
        'bin': '[TRASH BIN]',
        'instruction': 'Non-recyclable items'
    },
    'none': {
        'name': 'NO OBJECT DETECTED',
        'korean': 'Place item closer',
        'color': (100, 100, 100),
        'bin': '[WAITING]',
        'instruction': 'Place item in front of camera'
    }
}

def draw_ui_overlay(frame, category_info, confidence):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-80), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, verbose=False)
    probs = results[0].probs
    
    adjusted_probs = probs.data.clone()
    for idx, class_name in model.names.items():
        if class_name == 'paper':
            adjusted_probs[idx] *= PAPER_PENALTY
    
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    top_class_id = adjusted_probs.argmax().item()
    confidence = adjusted_probs[top_class_id].item()
    top_class_name = model.names[top_class_id]
    
    if confidence < CONFIDENCE_THRESHOLD:
        top_class_name = 'none'
        category = categories['none']
    else:
        category = categories.get(top_class_name, categories['trash'])
    
    frame = draw_ui_overlay(frame, category, confidence)
    
    h, w = frame.shape[:2]
    
    cv2.putText(frame, "SMART RECYCLING STATION", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"UWCD 5-Class Model | KNU Vision", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    box_w, box_h = 400, 180
    box_x, box_y = w - box_w - 20, 120
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), category['color'], 4)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, category['name'], (box_x + 15, box_y + 60), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, category['color'], 3)
    
    conf_text = f"Confidence: {confidence:.1%}"
    if confidence > 0.8:
        conf_color = (0, 255, 0)
    elif confidence > 0.6:
        conf_color = (0, 255, 255)
    else:
        conf_color = (0, 165, 255)
    
    cv2.putText(frame, conf_text, (box_x + 15, box_y + 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)
    
    cv2.putText(frame, f"{category['bin']}  |  {category['instruction']}", (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    if top_class_name == 'none':
        cv2.putText(frame, "READY TO SCAN", (w - 250, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    elif top_class_name != 'trash':
        cv2.putText(frame, "RECYCLABLE", (w - 200, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NON-RECYCLABLE", (w - 230, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Smart Recycling Station - UWCD Model", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nProgram terminated.")
