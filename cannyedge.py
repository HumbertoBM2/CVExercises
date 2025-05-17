import cv2 as cv
import numpy as np
import sys

def manualCanny(imgGray, threshLow, threshHigh):
    # Calcular gradientes con Sobel en direcciones X e Y
    gradX = cv.Sobel(imgGray, cv.CV_64F, 1, 0, ksize=3)
    gradY = cv.Sobel(imgGray, cv.CV_64F, 0, 1, ksize=3)
    # Obtener magnitud del gradiente y escalar a 0-255
    gradMag = np.sqrt(gradX**2 + gradY**2)
    gradMag *= 255.0 / gradMag.max()

    # Aplicar umbrales para extraer bordes
    edgeImg = np.zeros_like(gradMag, dtype=np.uint8)
    validMask = (gradMag >= threshLow) & (gradMag <= threshHigh)
    edgeImg[validMask] = 255
    return edgeImg

# Inicializar captura de video
videoCapture = cv.VideoCapture('chacarron.mp4')
if not videoCapture.isOpened():
    print("Error: no se pudo abrir el video.")
    sys.exit(1)

while True:
    hasFrame, frame = videoCapture.read()
    if not hasFrame:
        # Fin de los fotogramas o error de lectura
        print("Fin de los fotogramas. Saliendo.")
        break

    # Mostrar fotograma original
    cv.imshow('Video Original', frame)

    # Convertir a gris y aplicar detector de bordes manual
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    manualEdges = manualCanny(grayFrame, 30, 100)
    cv.imshow('Bordes Canny Manual', manualEdges)


    # Salir al presionar 'x'
    if cv.waitKey(5) & 0xFF == ord('x'):
        break

videoCapture.release()
cv.destroyAllWindows()
