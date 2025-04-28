import cv2
import numpy as np

# Carregar a imagem em escala de cinza
imagem = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)

harris = cv2.cornerHarris(imagem, blockSize=2, ksize=3, k=0.04)

harris = cv2.dilate(harris, None)

imagem[harris > 0.01 * harris.max()] = [255]

cv2.imshow("Harris Corner", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

