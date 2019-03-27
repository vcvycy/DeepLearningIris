import cv2
from matplotlib import pyplot as plt
def canny(img):
    edges = cv2.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def sobel(img):
    sobelXY = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize=3)
    return sobelXY

cv2.namedWindow("cjf")
img = r"E:\IrisDataset\2019.03.16_19.34.22_REYE_C320_250_R120.bmp"
# img = r"E:\IrisDataset\S5001R00.jpg"
img= cv2.imread(img,cv2.IMREAD_GRAYSCALE)
img= cv2.equalizeHist(img)

img =cv2.blur(img,(20,20))
cv2.imshow("cjf",img)
cv2.waitKey(0)
img = canny(img)
cv2.imshow("cjf",img)
cv2.waitKey(0)