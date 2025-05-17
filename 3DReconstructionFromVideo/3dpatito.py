import torch
import cv2
import numpy as np
import open3d as o3d

# Configuración inicial: ruta de la imagen
img_path = "imagenes3d/Patito.png"  
output_depth = "depth_map.jpg"
output_model = "pollito_1imagen.ply"

# Cargar el modelo MiDaS
print("🚀 Cargando modelo MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Preprocesamiento
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Leer la imagen
print(f"Procesando la imagen: {img_path}")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No se encontró la imagen en la ruta: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar transformaciones MiDaS
input_batch = midas_transforms(img_rgb)

# Ejecutar el modelo (predicción de profundidad)
print("Estimando mapa de profundidad...")
with torch.no_grad():
    prediction = midas(input_batch)  

# Redimensionar al tamaño original
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_map = prediction.cpu().numpy()

# Guardar el mapa de profundidad como imagen
depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(output_depth, depth_map_norm)
print(f"Mapa de profundidad guardado en {output_depth}")

# --- Reconstrucción 3D ---
print("🔧 Generando modelo 3D...")

h, w = depth_map.shape
fx = fy = w  # Aproximación sencilla (puedes ajustar si tienes datos exactos)
cx, cy = w / 2, h / 2

points = []
colors = []

# Para reducir tamaño, hacemos un paso de 2 píxeles
for v in range(0, h, 2):
    for u in range(0, w, 2):
        z = depth_map[v, u]
        if z == 0:
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append([x, y, z])
        colors.append(img[v, u] / 255.0)

points = np.array(points)
colors = np.array(colors)

# Crear la nube de puntos con Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Guardar la nube de puntos como archivo PLY
o3d.io.write_point_cloud(output_model, pcd)
print(f"Modelo 3D generado y guardado en {output_model}")

# Mostrar la nube de puntos (ventana interactiva)
o3d.visualization.draw_geometries([pcd])

print("La reconstrucción monocular está completa.")
