import cv2
import mediapipe as mp
import time

class HandMovementDetector:
    def __init__(
        self,
        cameraIndex=0,
        minDetectionConfidence=0.75,
        minTrackingConfidence=0.75,
        movementThreshold=20,
        areaChangeThreshold=1500,
        movementFile="handDir.txt",
        stateFile="hand_state.txt",
        stillFile="still_hand.txt",
        updateInterval=2.0,
        stillInterval=2.0
    ):
        # Abrir cámara con DirectShow probando varios índices
        for idx in range(cameraIndex, cameraIndex + 3):
            capture = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if capture.isOpened():
                self.capture = capture
                break
        else:
            raise IOError("No se pudo abrir la cámara en los índices probados.")

        # Configuración de MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            min_detection_confidence=minDetectionConfidence,
            min_tracking_confidence=minTrackingConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Estado previo para detección de movimiento
        self.prevXCenter = None
        self.prevYCenter = None
        self.prevArea = 0

        # Umbrales y archivos de salida
        self.movementThreshold = movementThreshold
        self.areaChangeThreshold = areaChangeThreshold
        self.movementFile = movementFile
        self.stateFile = stateFile
        self.stillFile = stillFile

        # Configuración de intervalos de actualización
        self.updateInterval = updateInterval
        self.stillInterval = stillInterval
        self.lastUpdateTime = 0
        self.lastMovementTime = time.time()
        self.stillMessageSent = False

    def writeMovement(self, direction):
        # Guardar la dirección de movimiento en archivo
        with open(self.movementFile, "w") as f:
            f.write(direction)

    def writeState(self, state):
        # Guardar el estado de la mano en archivo
        with open(self.stateFile, "w") as f:
            f.write(state)

    def detectXYMovement(self, currentXCenter, currentYCenter):
        # Detectar movimiento horizontal y vertical
        if self.prevXCenter is not None and self.prevYCenter is not None:
            deltaX = currentXCenter - self.prevXCenter
            deltaY = currentYCenter - self.prevYCenter
            direction = ""
            if abs(deltaX) > self.movementThreshold:
                direction = "Derecha" if deltaX > 0 else "Izquierda"
            if abs(deltaY) > self.movementThreshold:
                direction = "Abajo" if deltaY > 0 else "Arriba"
            if direction:
                print(direction)
                self.writeMovement(direction)
                self.lastMovementTime = time.time()
                self.stillMessageSent = False

    def detectZMovement(self, currentArea):
        # Detectar movimiento de profundidad basado en área del bounding box
        if self.prevArea != 0 and abs(currentArea - self.prevArea) > self.areaChangeThreshold:
            direction = "Adelante" if currentArea > self.prevArea else "Atras"
            print(direction)
            self.writeMovement(direction)
            self.lastMovementTime = time.time()
            self.stillMessageSent = False

    def calculateBoundingBox(self, landmarks, imgShape):
        # Calcular centro y área del bounding box usando puntos clave de la mano
        xMin, yMin = float("inf"), float("inf")
        xMax, yMax = 0, 0
        for idx in [0, 1, 5, 9, 13]:
            lm = landmarks[idx]
            x, y = int(lm.x * imgShape[1]), int(lm.y * imgShape[0])
            xMin, xMax = min(x, xMin), max(x, xMax)
            yMin, yMax = min(y, yMin), max(y, yMax)
        centerX = (xMin + xMax) // 2
        centerY = (yMin + yMax) // 2
        area = (xMax - xMin) * (yMax - yMin)
        boundingBox = (xMin, yMin, xMax, yMax)
        return centerX, centerY, area, boundingBox

    def isHandOpen(self, landmarks):
        # Verificar si la mano está abierta contando dedos extendidos
        tipIds = [4, 8, 12, 16, 20]
        knuckleIds = [2, 6, 10, 14, 18]
        openFingers = 0
        for tipId, knuckleId in zip(tipIds, knuckleIds):
            if landmarks[tipId].y < landmarks[knuckleId].y:
                openFingers += 1
        return openFingers >= 4

    def detectMovement(self):
        # Bucle principal de captura y detección
        try:
            while True:
                success, frame = self.capture.read()
                if not success:
                    print("No se pudo capturar el frame")
                    break

                rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgbFrame)

                if results.multi_hand_landmarks:
                    for handLandmarks in results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(
                            frame, handLandmarks, self.mpHands.HAND_CONNECTIONS
                        )
                        centerX, centerY, area, boundingBox = self.calculateBoundingBox(
                            handLandmarks.landmark, frame.shape
                        )
                        cv2.rectangle(
                            frame,
                            (boundingBox[0], boundingBox[1]),
                            (boundingBox[2], boundingBox[3]),
                            (0, 255, 0),
                            2
                        )

                        self.detectXYMovement(centerX, centerY)
                        self.detectZMovement(area)

                        currentTime = time.time()
                        if currentTime - self.lastUpdateTime >= self.updateInterval:
                            if self.isHandOpen(handLandmarks.landmark):
                                cv2.putText(
                                    frame, "Mano Abierta", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                                )
                                self.writeState("Abierta")
                                print("Mano Abierta")
                            else:
                                cv2.putText(
                                    frame, "Mano Cerrada", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                                )
                                self.writeState("Cerrada")
                                print("Mano Cerrada")
                            self.lastUpdateTime = currentTime

                        self.prevXCenter = centerX
                        self.prevYCenter = centerY
                        self.prevArea = area

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandMovementDetector()
    detector.detectMovement()
