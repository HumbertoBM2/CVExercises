# 🦆 TRACKING DE PATITOS

# ✅ IMPORTS
import sys
import cv2
import numpy as np
from collections import defaultdict
from roboflow import Roboflow
from sort import Sort

# ✅ PARÁMETROS DE TRACKER (ajustados)
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.2)
trajectories_by_id = defaultdict(list)

# ✅ CONFIGURACIÓN DE VIDEO
video_path = "pollitos.mp4"

# ✅ CONEXIÓN A ROBOFLOW
rf = Roboflow(api_key="WG2KIDImNzEzMLxmVznj")
project = rf.workspace("patitosmuoz").project("duck_tracker")
model = project.version(1).model

# ✅ CALIBRACIÓN
# Cada cuadro del piso mide 30x30 cm, ajustamos la relación pixels/cm
# Esto puede necesitar calibrarse mejor si sabes cuántos píxeles mide una baldosa en tu video
cm_per_pixel = 30 / 100  # SUPONIENDO que ~100 px = 30 cm

# ✅ PROCESAR VIDEO
cap = cv2.VideoCapture(video_path)
frame_count = 0

print("🔄 Procesando video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Hacer predicción
    results = model.predict(frame, confidence=40, overlap=30).json()

    # ✅ Formatear detecciones para SORT: [x1, y1, x2, y2, confidence]
    detections = []
    for pred in results['predictions']:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        conf = pred.get('confidence', 0.9)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        detections.append([x1, y1, x2, y2, conf])

    # ✅ SORT necesita (0,5) shape aunque esté vacío
    if len(detections) == 0:
        detections = np.empty((0, 5))
    else:
        detections = np.array(detections)

    # ✅ Actualizar tracker
    tracked_objects = tracker.update(detections)

    # ✅ Guardar trayectoria por ID
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        trajectories_by_id[int(track_id)].append((cx, cy, frame_count))

        # Dibujar caja y texto
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    frame_count += 1

cap.release()
print("✅ Procesamiento terminado.")

# ✅ ANALIZAR DISTANCIAS POR PATO
print("\n📊 Resultados:")

for track_id, traj in trajectories_by_id.items():
    if len(traj) < 15:
        print(f"🦆 ID {track_id} ignorado (solo {len(traj)} puntos)")
        continue

    print(f"🦆 ID {track_id} tiene {len(traj)} puntos guardados")

    points = np.array(traj, dtype=float)
    diffs = np.diff(points[:, :2], axis=0)
    distances_pixels = np.linalg.norm(diffs, axis=1)
    distances_cm = distances_pixels * cm_per_pixel
    total_cm = np.sum(distances_cm)
    print(f"🦆 ID {track_id}: {total_cm:.2f} cm recorridos")
