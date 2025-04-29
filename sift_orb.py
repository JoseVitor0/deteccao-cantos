import cv2
import numpy as np


imagem = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
sift_keypoints, _ = sift.detectAndCompute(imagem, None)
sift_result = cv2.drawKeypoints(imagem, sift_keypoints, None)

# ORB
orb = cv2.ORB_create()
orb_keypoints, _ = orb.detectAndCompute(imagem, None)
orb_result = cv2.drawKeypoints(imagem, orb_keypoints, None, color=(0, 255, 0))

# Exibir resultados lado a lado
cv2.imshow('SIFT', sift_result)
cv2.imshow('ORB', orb_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
