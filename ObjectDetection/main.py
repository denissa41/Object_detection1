import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==============================
# MODEL
# ==============================
model_laptop = YOLO("yolov8n.pt")
model_phone = YOLO("yolov8n.pt")

# ==============================
# CAMERE
# ==============================
cap_laptop = cv2.VideoCapture(0)
cap_phone = cv2.VideoCapture(1)

for cap in [cap_laptop, cap_phone]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==============================
# GLOBAL ID SYSTEM
# ==============================
global_id_counter = 0
global_objects = {}  # (class + local_id + camera) -> global_id

def get_global_id(camera, local_id, cls):
    global global_id_counter, global_objects

    key = (camera, local_id, cls)

    if key in global_objects:
        return global_objects[key]

    global_id_counter += 1
    global_objects[key] = global_id_counter
    return global_id_counter

# ==============================
# LABELS
# ==============================
clase_ro = {
    "person": "Client",
    "backpack": "Rucsac",
    "handbag": "Geanta",
    "bottle": "Sticla",
    "cell phone": "Telefon",
    "book": "Carte",
    "cup": "Pahar",
    "laptop": "Laptop"
}

# ==============================
# DETECT FUNCTION
# ==============================
def process(frame, camera_name, model, color):
    frame = cv2.resize(frame, (640, 360))

    results = model.track(
        frame,
        persist=True,
        conf=0.3,
        imgsz=640,
        tracker="bytetrack.yaml",
        verbose=False
    )

    count = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        count = len(ids)

        for box, local_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box

            label_en = model.names[cls]
            label_ro = clase_ro.get(label_en, label_en)

            global_id = get_global_id(camera_name, local_id, label_en)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{label_ro} GID:{global_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

    cv2.putText(frame, camera_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    return frame, count

# ==============================
# LOOP
# ==============================
print("=== MULTI CAMERA GLOBAL ID SYSTEM ===")

while True:
    ret1, frame1 = cap_laptop.read()
    ret2, frame2 = cap_phone.read()

    if not ret1 and not ret2:
        break

    if ret1:
        frame1, c1 = process(frame1, "LAPTOP", model_laptop, (255, 0, 0))
    else:
        frame1 = np.zeros((360, 640, 3), dtype=np.uint8)

    if ret2:
        frame2, c2 = process(frame2, "PHONE", model_phone, (0, 255, 0))
    else:
        frame2 = np.zeros((360, 640, 3), dtype=np.uint8)

    combined = cv2.hconcat([frame1, frame2])

    # INFO GLOBAL
    cv2.putText(combined,
                f"GLOBAL OBJECTS: {global_id_counter}",
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.putText(combined,
                "AI MULTI CAMERA SYSTEM (GLOBAL ID)",
                (200, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.imshow("AI System", combined)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap_laptop.release()
cap_phone.release()
cv2.destroyAllWindows()