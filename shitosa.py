import cv2
import numpy as np



imagem = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)


cantos = cv2.goodFeaturesToTrack(imagem, maxCorners=500, qualityLevel=0.05, minDistance=10)
cantos = np.int32(cantos)

# Marcar os cantos na imagem
for canto in cantos:
    x, y = canto.ravel()
    cv2.circle(imagem, (x, y), 3, 255, -1)

# Exibir resultados
cv2.imshow('Shi-Tomasi', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()