import cv2
import os
import time
import numpy as np

from Cropping import ExtractAndStraightenFromImage
from LocateGrid import DetectGrid
from LocateGrid import morph_gradient
from LocateGrid import bit_masking_grad
from LocateGrid import adaptive_thresh

IMAGE_FILE_PATH = os.path.join("Capture", "BoardPictures")

# Create directory to save pictures if it doesn't exist
if not os.path.exists(IMAGE_FILE_PATH):
    os.makedirs(IMAGE_FILE_PATH)

# Start the video capture
vid = cv2.VideoCapture(0)  # ID 1 assumes a second camera (like your Orbbec Astra). Use 0 for default camera

is_automatic = False


while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break

    key = cv2.waitKey(1)

    boardImg = ExtractAndStraightenFromImage(frame)
    checkBoard = DetectGrid(boardImg)
    adapt_img = cv2.rotate(adaptive_thresh(boardImg), cv2.ROTATE_90_COUNTERCLOCKWISE)
    morphImg = morph_gradient(boardImg)
    bitResult = bit_masking_grad(morphImg, boardImg)


    cv2.imshow("Frame", frame)
    #cv2.imshow("Board img", boardImg)

    cv2.imshow("Check board", checkBoard)
    cv2.imshow("Adaptive Thres Method", adapt_img)
    cv2.imshow("Bit Result", bitResult)

    if key & 0xFF == ord('q'):
        break
            

vid.release()
cv2.destroyAllWindows()
