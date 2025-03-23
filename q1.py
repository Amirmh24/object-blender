import numpy as np
import cv2
from scipy.sparse import lil_matrix,linalg


def makeBorder(img):
    hei, wid = img.shape
    imgBrd = img.copy()
    global borderColor
    img[0, :], img[hei-1, :], img[:, 0], img[:, wid-1] = 0, 0, 0, 0
    for i in range(hei):
        for j in range(wid):
            if (img[i, j] ==255 ):
                if (img[i - 1, j] == 0): imgBrd[i - 1, j] = borderColor
                if (img[i + 1, j] == 0): imgBrd[i + 1, j] = borderColor
                if (img[i, j - 1] == 0): imgBrd[i, j - 1] = borderColor
                if (img[i, j + 1] == 0): imgBrd[i, j + 1] = borderColor
    return imgBrd


def laplacian(img, i, j):
    return 4 * img[i, j] - img[i - 1, j] - img[i + 1, j] - img[i, j - 1] - img[i, j + 1]


def getXMatrix(imgSrc, imgTrg, imgMsk):
    indicesLoc={}
    index=0
    for i in range(imgMsk.shape[0]):
        for j in range(imgMsk.shape[1]):
            if(imgMsk[i,j]!=0):
                indicesLoc[(i,j)]=index
                index=index+1
    A = lil_matrix((len(indicesLoc), len(indicesLoc)),dtype=float)
    b = lil_matrix((len(indicesLoc),1),dtype=float)
    for i,j in indicesLoc:
        n=indicesLoc[(i,j)]
        if (imgMsk[i, j] == borderColor):
            b[n] = imgTrg[i, j]
            A[n, n] = 1
        else:
            b[n] = laplacian(imgSrc, i, j)
            A[n, n] = 4
            m = indicesLoc[(i-1,j)]
            A[n, m] = -1
            m = indicesLoc[(i+1,j)]
            A[n, m] = -1
            m = indicesLoc[(i,j-1)]
            A[n, m] = -1
            m = indicesLoc[(i,j+1)]
            A[n, m] = -1
    X = linalg.spsolve(A,b)
    return X

borderColor = 100
Isrc = cv2.imread("1.source.jpg")
height, width, channels = Isrc.shape
Itrg = cv2.imread("2.target.jpg")
Ians = Itrg.copy()
Imsk = cv2.imread("mask1.jpg")
Imsk = cv2.cvtColor(Imsk, cv2.COLOR_BGR2GRAY)
Imsk = cv2.threshold(Imsk, 100, 255, cv2.THRESH_BINARY)[1]
Imsk = makeBorder(Imsk)
x, y = 80,20
imask, jmask = np.where(Imsk != 0)
for c in range(channels):
    X = getXMatrix(Isrc[:, :, c], Itrg[x:x + height, y:y + width, c], Imsk)
    for n in range(len(X)):
        Ians[imask[n] + x, jmask[n] + y, c] = max(min(X[n], 255), 0)
cv2.imwrite("res1.jpg", Ians)
