import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ✅ CONFIGURACIÓN DE VIDEO
video_path = "pollitos.mp4"

# ✅ PARAMS PARA FLOW
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# ✅ INICIALIZAR CAPTURA
cap = cv2.VideoCapture(video_path)

# ✅ LEER PRIMER FRAME
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ✅ DETECTAR PUNTOS INICIALES
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

# ✅ VARIABLES PARA GUARDAR MOVIMIENTO
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
        # 🔄 Si quedan pocos puntos, detectar nuevos
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if prev_pts is None:
            print(f"⚠️ No se pudieron detectar nuevos puntos en frame {frame_count}.")
            prev_gray = gray.copy()
            frame_count += 1
            continue

    # ✅ Calcular Optical Flow
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    if next_pts is None:
        print(f"⚠️ No se pudo calcular Optical Flow en frame {frame_count}. Saltando frame.")
        prev_gray = gray.copy()
        frame_count += 1
        continue

    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) == 0 or len(good_next) == 0:
        print(f"⚠️ No hay puntos válidos en frame {frame_count}. Saltando frame.")
        prev_gray = gray.copy()
        frame_count += 1
        continue

    # ✅ Estimar traslación (promedio del movimiento)
    movement = good_next - good_prev
    dx = np.mean(movement[:, 0])
    dy = np.mean(movement[:, 1])
    displacements.append((dx, dy))

    # ✅ Estimar rotación
    angles_prev = np.arctan2(good_prev[:, 1] - np.mean(good_prev[:, 1]),
                             good_prev[:, 0] - np.mean(good_prev[:, 0]))
    angles_next = np.arctan2(good_next[:, 1] - np.mean(good_next[:, 1]),
                             good_next[:, 0] - np.mean(good_next[:, 0]))
    dtheta = np.mean(angles_next - angles_prev)
    rotations.append(dtheta)

    frames_idx.append(frame_count)

    # Preparar siguiente iteración
    prev_gray = gray.copy()
    prev_pts = good_next.reshape(-1, 1, 2)

    frame_count += 1

cap.release()
print("✅ Análisis terminado.")

# ✅ CONVERTIR A CM
cm_per_pixel = 30 / 100
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

print("\n✅ Gráficas guardadas: cameraman_trayectoria_2D.png, cameraman_trayectoria_3D.png, cameraman_giro_por_frame.png")
