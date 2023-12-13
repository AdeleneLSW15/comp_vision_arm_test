import cv2
import numpy as np
from matplotlib import pyplot as plt

def DetectGrid(boardImg):
    img = _ConvertToBinary(boardImg)

    return img

def _ConvertToBinary(boardImg):
    # Convert to grayscale
    gray = cv2.cvtColor(boardImg, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image (adjust the threshold value as needed)
    _, binary = cv2.threshold(gray, 215/2, 215, cv2.THRESH_BINARY)  # 200 is the threshold, adjust if necessary
    #1 is 255 binary. Therefore that is the maximum resolution.
    #255/2, 255
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #implementig adaptive thresholding
    # perform adaptive thresholding again, this time using a Gaussian
    # weighting versus a simple mean to compute our local threshold
    # value
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    cv2.imshow("Gaussian Adaptive Thresholding", thresh)


    return opening


#code to cut up square regions and then stitch them back together.
def get_square_regions(img):
    #Apply Canny Edge detection
    #if image is none -> throws an error
    assert img is not None, "file can't be read, check path exists"
    #cv.Canny(img_source, minVal, maxVal)
    edges = cv2.Canny(img, 100,200) 
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Original Image'), plt.xticks[1]

#trying to improve resolution of the chess pieces to enable piece detection.
#Using Dilation and Erosion methods (testing).
#Dilation and Erosion doesn't work well for pre-binary and post-binary. Move on to other methods!
def chess_dilation(img):
    kernel = np.ones((5,5), np.uint8)
    
    
    img_erosion = cv2.erode(img, kernel, iterations = 1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations = 1)

    return img_dilation

#adaptive thres image output works well with circle clearly visible. To do -> implement slider for different light conditions.
def adaptive_thresh(img):
    #works well, getting circle.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    return adapt

#Try top hat, morphological gradient method. Morph gradient outputs black white overlay
# try bit masking over this layer.
def morph_gradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    #set img as boardchess in MAIN.
    #invert = cv2.bitwise_not(img)
    morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations = 1) 
    return morph_gradient

#on uneven light condition - some pieces is not clearly displayed even with bitmasking
#Try: Adaptive thresholding to resolve uneven lighting condition.
def bit_masking_grad(img_grad, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the gradient to create a binary mask
    _, binary_mask = cv2.threshold(img_grad, 130/2, 255, cv2.THRESH_BINARY)
    #30 works well, try different thresholding
    result = cv2.bitwise_and(gray, gray, mask=binary_mask)
    return result

#Also: think about what to do when there is obstacles on chess board => e.g. robot arm moving


#Try out template matching for chess piece detection.
