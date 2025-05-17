import cv2
import numpy as np
from collections import defaultdict
from roboflow import Roboflow
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################
# 🦆 PARTE 1: TRACKING DE PATITOS + GRAFICAS #
##############################################

print("\n========== 🦆 TRACKING DE PATITOS ==========")

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
video_path = "pollitos.mp4"

# ✅ CONEXIÓN A ROBOFLOW
rf = Roboflow(api_key="WG2KIDImNzEzMLxmVznj")
project = rf.workspace("patitosmuoz").project("duck_tracker")
model = project.version(1).model

# ✅ CALIBRACIÓN (ajusta si es necesario)
cm_per_pixel = 30 / 100

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

    if frame_count % 100 == 0:
        print(f"Procesados {frame_count} frames...")

    frame_count += 1

cap.release()
print("✅ Procesamiento terminado.")
print("\n📊 Resultados (Patitos):")

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
    plt.savefig(f'pato_{new_id}_trayectoria_2D.png')
    plt.close()

    # ✅ Graficar 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0] * cm_per_pixel, points[:, 1] * cm_per_pixel, points[:, 2], marker='o')
    ax.set_title(f'Trayectoria 3D - Pato {new_id}')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Frame')
    plt.savefig(f'pato_{new_id}_trayectoria_3D.png')
    plt.close()

##############################################
# 📷 PARTE 2: MOVIMIENTO DEL CAMARÓGRAFO     #
##############################################

print("\n========== 📷 MOVIMIENTO DEL CAMARÓGRAFO ==========")

# ✅ PARAMS PARA FLOW
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

cap = cv2.VideoCapture(video_path)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

displacements = []
rotations = []
frames_idx = []

frame_count = 0

print("🔄 Analizando movimiento del camarógrafo...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is None or len(prev_pts) < 10:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if prev_pts is None:
            prev_gray = gray.copy()
            frame_count += 1
            continue

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    if next_pts is None:
        prev_gray = gray.copy()
        frame_count += 1
        continue

    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) == 0 or len(good_next) == 0:
        prev_gray = gray.copy()
        frame_count += 1
        continue

    movement = good_next - good_prev
    dx = np.mean(movement[:, 0])
    dy = np.mean(movement[:, 1])
    displacements.append((dx, dy))

    angles_prev = np.arctan2(good_prev[:, 1] - np.mean(good_prev[:, 1]),
                             good_prev[:, 0] - np.mean(good_prev[:, 0]))
    angles_next = np.arctan2(good_next[:, 1] - np.mean(good_next[:, 1]),
                             good_next[:, 0] - np.mean(good_next[:, 0]))
    dtheta = np.mean(angles_next - angles_prev)
    rotations.append(dtheta)

    frames_idx.append(frame_count)

    prev_gray = gray.copy()
    prev_pts = good_next.reshape(-1, 1, 2)
    frame_count += 1

cap.release()
print("✅ Análisis terminado.")

# ✅ CONVERTIR A CM
displacements_cm = np.array(displacements) * cm_per_pixel

# ✅ DISTANCIA TOTAL
distances = np.linalg.norm(displacements_cm, axis=1)
total_distance = np.sum(distances)
print(f"\n📏 Desplazamiento total estimado: {total_distance:.2f} cm")

# ✅ GIRO TOTAL
rotations_deg = np.degrees(rotations)
total_rotation = np.sum(rotations_deg)
print(f"🔄 Giro total estimado: {total_rotation:.2f} grados")

# ✅ GRAFICAR TRAYECTORIA 2D
trajectory = np.cumsum(displacements_cm, axis=0)
plt.figure()
plt.title("Trayectoria 2D del camarógrafo")
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.grid()
plt.savefig("cameraman_trayectoria_2D.png")
plt.close()

# ✅ GRAFICAR TRAYECTORIA 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], frames_idx, marker='o')
ax.set_title("Trayectoria 3D del camarógrafo")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Frame")
plt.savefig("cameraman_trayectoria_3D.png")
plt.close()

# ✅ GRAFICAR GIRO
plt.figure()
plt.title("Giro del camarógrafo (por frame)")
plt.plot(frames_idx, rotations_deg, marker='o')
plt.xlabel("Frame")
plt.ylabel("Δ Ángulo (°)")
plt.grid()
plt.savefig("cameraman_giro_por_frame.png")
plt.close()

print("\n✅ Gráficas guardadas: todas las gráficas de los patitos + las del camarógrafo 🚀")
