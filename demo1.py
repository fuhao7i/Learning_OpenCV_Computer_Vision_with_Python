import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread("/Users/fuhao7i/Desktop/AJu/1.JPG")
cv.namedWindow("Image",0)
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows()
