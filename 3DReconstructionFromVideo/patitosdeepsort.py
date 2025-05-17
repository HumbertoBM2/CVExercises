# 🦆 TRACKING DE PATITOS CON DEEPSORT (USANDO GPU) - SOLO 7 PATOS + GRAFICAS

import cv2
import numpy as np
from collections import defaultdict
from roboflow import Roboflow
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ✅ CHECAR GPU DISPONIBLE
if torch.cuda.is_available():
    print(f"✅ Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No se detectó GPU. Usando CPU.")

# ✅ INICIALIZAR DEEPSORT
tracker = DeepSort(
    max_age=15,
    n_init=3,
    max_cosine_distance=0.3,
    half=True,
    bgr=True,
    embedder_gpu=True
)

trajectories_by_id = defaultdict(list)

# ✅ CONFIGURACIÓN DE VIDEO
video_path = "pollitos.mp4"

# ✅ CONEXIÓN A ROBOFLOW
rf = Roboflow(api_key="WG2KIDImNzEzMLxmVznj")
project = rf.workspace("patitosmuoz").project("duck_tracker")
model = project.version(1).model

# ✅ CALIBRACIÓN (ajusta si es necesario)
cm_per_pixel = 30 / 100

# ✅ PROCESAR VIDEO
cap = cv2.VideoCapture(video_path)
frame_count = 0

print("🔄 Procesando video con DeepSORT + GPU...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Hacer predicción con Roboflow
    results = model.predict(frame, confidence=40, overlap=30).json()

    detections = []

    for pred in results['predictions']:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        conf = pred.get('confidence', 0.9)

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        detections.append([[x1, y1, x2, y2], conf])

    # ✅ Actualizar tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # ✅ Guardar trayectorias por ID
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = ltrb
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        trajectories_by_id[int(track_id)].append((cx, cy, frame_count))

    # Mostrar progreso
    if frame_count % 100 == 0:
        print(f"Procesados {frame_count} frames...")

    frame_count += 1

cap.release()
print("✅ Procesamiento terminado.")
print("\n📊 Resultados:")

sorted_tracks = sorted(trajectories_by_id.items(), key=lambda x: len(x[1]), reverse=True)
top_7_tracks = sorted_tracks[:7]

for new_id, (original_id, traj) in enumerate(top_7_tracks, start=1):
    print(f"🦆 Pato {new_id} (Original ID {original_id}) tiene {len(traj)} puntos guardados")

    points = np.array(traj, dtype=float)
    diffs = np.diff(points[:, :2], axis=0)
    distances_pixels = np.linalg.norm(diffs, axis=1)
    distances_cm = distances_pixels * cm_per_pixel
    total_cm = np.sum(distances_cm)
    print(f"🦆 Pato {new_id}: {total_cm:.2f} cm recorridos")

    # ✅ Graficar 2D
    plt.figure()
    plt.title(f'Trayectoria 2D - Pato {new_id}')
    plt.plot(points[:, 0] * cm_per_pixel, points[:, 1] * cm_per_pixel, marker='o', markersize=3)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.grid()
    plt.savefig(f'pato_{new_id}_trayectoria_2D.png')  # Guarda la gráfica como imagen
    plt.close()

    # ✅ Graficar 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0] * cm_per_pixel, points[:, 1] * cm_per_pixel, points[:, 2], marker='o')
    ax.set_title(f'Trayectoria 3D - Pato {new_id}')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Frame')
    plt.savefig(f'pato_{new_id}_trayectoria_3D.png')  # Guarda la gráfica como imagen
    plt.close()
