import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def detectarEsquinas(imagenGris, maxCorners=20, qualityLevel=0.01, minDistance=30):
    """Detecta esquinas relevantes en escala de grises."""
    puntos = cv2.goodFeaturesToTrack(
        imagenGris,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance
    )
    return np.squeeze(puntos).astype(np.float64)

class RobustICP3D2D:
    def __init__(self, modelPoints3D, focalLength, imageSize):
        """Configura el modelo 3D y los parámetros de la cámara."""
        self.model3D     = modelPoints3D.astype(np.float64)
        self.focalLength = float(focalLength)
        self.centerX     = imageSize[0] / 2.0
        self.centerY     = imageSize[1] / 2.0
        self.edgesList   = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]

    def projectPoints(self, rotMat, transVec, scale=np.ones(3)):
        """Aplica proyección perspectiva al conjunto de vértices 3D."""
        puntos3D = (rotMat @ (self.model3D * scale).T).T + transVec
        x2d      = puntos3D[:,0] * self.focalLength / puntos3D[:,2] + self.centerX
        y2d      = puntos3D[:,1] * self.focalLength / puntos3D[:,2] + self.centerY
        return np.column_stack((x2d, y2d)), puntos3D

    def findClosestPoints(self, imgPts2D, rotMat, transVec, scale):
        """Encuentra correspondencias 3D←→2D mediante búsqueda de vecinos."""
        proy2D, cam3D = self.projectPoints(rotMat, transVec, scale)
        indices      = KDTree(imgPts2D).query(proy2D)[1]
        matched2D    = imgPts2D[indices]
        rayos        = np.column_stack([
            matched2D[:,0] - self.centerX,
            matched2D[:,1] - self.centerY,
            np.full(len(matched2D), self.focalLength)
        ])
        rayos       /= np.linalg.norm(rayos, axis=1)[:,None]
        lambdas      = np.einsum('ij,ij->i', cam3D, rayos)
        return cam3D, rayos * lambdas[:,None]

    def estimateRigidAndScale(self, src3D, dst3D, weights=None):
        """Calcula rotación, traslación y escala entre dos nubes 3D."""
        if weights is None:
            weights = np.ones(len(src3D))
        centroSrc = np.average(src3D, axis=0, weights=weights)
        centroDst = np.average(dst3D, axis=0, weights=weights)
        srcC = src3D - centroSrc
        dstC = dst3D - centroDst
        H    = (srcC * weights[:,None]).T @ dstC
        U,_,Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t    = centroDst - R @ centroSrc
        srcRot = (R @ srcC.T).T
        num    = (weights * np.sum(dstC * srcRot, axis=1)).sum()
        den    = (weights * np.sum(srcRot**2, axis=1)).sum()
        s      = num / den
        return R, t, np.array([s, s, s])

    def optimizeRegistration(self, rotInit, transInit, imgPts2D, maxIter=50000):
        """Refina la pose con ICP robusto usando ponderación Huber."""
        R, t, scale = rotInit.copy(), transInit.copy(), np.ones(3)
        historyErr, historyScale = [], []

        for _ in range(maxIter):
            src3D, dst3D = self.findClosestPoints(imgPts2D, R, t, scale)
            residuals    = np.linalg.norm(src3D - dst3D, axis=1)
            sigma        = max(1e-12, 1.4826 * np.median(residuals))
            weights      = np.where(
                residuals > sigma,
                2*sigma/(2*residuals + 1e-8),
                1.0
            )
            R_u, t_u, s_u = self.estimateRigidAndScale(src3D, dst3D, weights)
            R, t, scale   = R_u @ R, R_u @ t + t_u, scale * s_u

            errMean = residuals.mean()
            historyErr.append(errMean)
            historyScale.append(scale[0])

            if errMean < 1e-15:
                break
            if len(historyErr) > 1 and abs(historyErr[-1] - historyErr[-2]) < 1e-9:
                break

        return R, t, scale, historyErr, historyScale

    def visualizeResults(self, grayImage, imgPts2D,
                         rotInit, transInit, scaleInit,
                         rotFinal, transFinal, scaleFinal,
                         historyErr, historyScale):
        """Presenta la imagen original, el resultado alineado y las gráficas."""
        _, _ = self.projectPoints(rotInit,  transInit,  scaleInit)   # proyección inicial omitida
        proyFinal2D, _ = self.projectPoints(rotFinal, transFinal, scaleFinal)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(grayImage, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title("Original")
        axs[1].imshow(grayImage, cmap='gray')
        axs[1].axis('off')
        axs[1].set_title("Alineado")

        for a, b in self.edgesList:
            p, q = proyFinal2D[a], proyFinal2D[b]
            axs[1].plot([p[0], q[0]], [p[1], q[1]], '-r')
        axs[1].scatter(imgPts2D[:,0], imgPts2D[:,1], c='b', s=30)

        plt.tight_layout()
        plt.show()

        fig2, (axErr, axScale) = plt.subplots(2, 1, figsize=(6, 8))
        axErr.plot(historyErr)
        axErr.set_ylabel("Error medio")
        axErr.grid(True)
        axScale.plot(historyScale)
        axScale.set_ylabel("Escala")
        axScale.set_xlabel("Iteración")
        axScale.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    img       = cv2.imread("cube.png")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgPts2D  = detectarEsquinas(grayImage)

    baseCube  = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]
    ], float)
    model3D   = (baseCube - 0.5) * 180.0

    rotInit, _ = cv2.Rodrigues(np.random.uniform(-0.1, 0.1, 3))
    transInit  = np.array([50.0, -40.0, 1600.0])

    icp = RobustICP3D2D(model3D, focalLength=1500.0, imageSize=grayImage.shape[::-1])
    rotFinal, transFinal, scaleFinal, historyErr, historyScale = icp.optimizeRegistration(
        rotInit, transInit, imgPts2D
    )

    icp.visualizeResults(
        grayImage, imgPts2D,
        rotInit, transInit, np.ones(3),
        rotFinal, transFinal, scaleFinal,
        historyErr, historyScale
    )
