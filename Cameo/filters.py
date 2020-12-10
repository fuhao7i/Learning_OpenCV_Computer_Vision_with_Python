import cv2
import numpy
import utils

def strokeEdges(src, dst, blurKsize = 5, edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    
    """cv2.CV_8U表示位深度，每个通道为8位，如果为-1的话就表示目标图像和愿图像有同样的位深度"""
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize= edgeKsize)
    """
        特殊的高通滤波器：
            通过对图像应用低通滤波器之后，与原始图像进行差值。
    """
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

class VconvolutionFilter(object):

    def __init__(self, kernel):
        self._kernel = kernel
    
    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VconvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
        VconvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(VconvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        VconvolutionFilter.__init__(self, kernel)

# 邻近平均滤波器
class BlurFilter(VconvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])

        VconvolutionFilter.__init__(self, kernel)

class EmbossFilter(VconvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VconvolutionFilter.__init__(self, kernel)
