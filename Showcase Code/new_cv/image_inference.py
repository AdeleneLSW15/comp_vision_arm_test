import cv2
import os
import time
import numpy as np
import csv

from Cropping import ExtractAndStraightenFromImage
from LocateGrid import DetectGrid
from LocateGrid import morph_gradient
from LocateGrid import bit_masking_grad
from LocateGrid import adaptive_thresh

IMAGE_FILE_PATH = os.path.join("Capture", "BoardPictures")
CSV_FILE_PATH = "points.csv"  # Specify the CSV file path

# Create directory to save pictures if it doesn't exist
if not os.path.exists(IMAGE_FILE_PATH):
    os.makedirs(IMAGE_FILE_PATH)

# Open the CSV file for writing
with open(CSV_FILE_PATH, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["x", "y"])  # Write the header row


# Start the video capture
vid = cv2.VideoCapture(0)  # ID 1 assumes a second camera (like your Orbbec Astra). Use 0 for default camera

is_automatic = False
count = 0

#Extract chessboard squares method, add it to screenshot.
def extract_chessboard_squares(image_path, rows, columns):
	#take inner corners by subtracting 1 from provided values of rows and columns.
	#as there is "missing corners" at the edge of the board (not intersection point).
    inner_rows = rows - 1
    inner_columns = columns - 1
    # Load the chessboard image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard inner corners -> find the number of inner corners to be provided!
    found, corners = cv2.findChessboardCorners(gray, (inner_rows, inner_columns), None)

    if found:
        # Refine the corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # Create a 2D list to store file names of chessboard squares
        square_files = [[None] * (columns-2) for _ in range(rows-2)]

        # Iterate over each chessboard square
        for row in range(rows-2):
            for col in range(columns-2):
                # Calculate the square corners' coordinates
                pt1 = corners[row * inner_columns + col]
                pt2 = corners[(row + 1) * inner_columns + col]
                pt3 = corners[(row + 1) * inner_columns + col + 1]
                pt4 = corners[row * inner_columns + col + 1]

                # Create a mask for the current square
                mask = np.zeros_like(image)
                cv2.fillConvexPoly(mask, np.int32([pt1, pt2, pt3, pt4]), (255, 255, 255))

                # Apply the mask to the original image
                masked_image = cv2.bitwise_and(image, mask)

                # Save the masked image to a temporary file
                _, temp_filename = tempfile.mkstemp(suffix='.jpg')
                cv2.imwrite(temp_filename, masked_image)

                # Store the temporary file name in the 2D list
                square_files[row][col] = temp_filename

        return square_files

    else:
        print("Chessboard not found.")
        return None

def screenshot(img_to_take):
    global count

    if cv2.waitKey(1) & 0xFF == ord('v'):
        cv2.imwrite(os.path.join(os.getcwd(), 'image_capture', f'{count}.png'), img_to_take)
        count += 1

        # Process the screenshot and save corner detections to CSV
        inference_img = cv2.imread(os.path.join(os.getcwd(), 'image_capture', '0.png'))
        gray = cv2.cvtColor(inference_img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 5, 0.06)
        dst = cv2.dilate(dst, None)

        # Find strong corner points and save their coordinates to the CSV file
        for y in range(dst.shape[0]):
            for x in range(dst.shape[1]):
                if dst[y, x] > 0.01 * dst.max():
                    with open(CSV_FILE_PATH, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([x, y])

"""def screenshot(img_to_take):
    global count
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    
    
    if cv2.waitKey(1) & 0xFF == ord('v'):
        #print("C - capture image")
        #print(os.path.join(os.getcwd(), path_output_dir, f'{count}.png'))
        cv2.imwrite(os.path.join(os.getcwd(), 'image_capture', f'{count}.png'), img_to_take)
        count += 1

        
    # cv2.destroyAllWindows()
    # vid.release()
"""

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break

    key = cv2.waitKey(1)

    boardImg = ExtractAndStraightenFromImage(frame)
    checkBoard = DetectGrid(boardImg)
    adapted_feed = cv2.rotate(adaptive_thresh(boardImg), cv2.ROTATE_90_COUNTERCLOCKWISE)
    morphImg = morph_gradient(boardImg)
    bitResult = bit_masking_grad(morphImg, boardImg)

    screenshot(checkBoard)
    # ------------------------------------------------------

    #simple rectangle detection doesn't work well - try Harris Corner Detection + Denoising from light
    if os.path.exists(os.path.join(os.getcwd(), 'image_capture', '0.png')):

        inference_img = cv2.imread(os.path.join(os.getcwd(), 'image_capture', '0.png'))
        gray = cv2.cvtColor(inference_img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,5,0.06)
        dst = cv2.dilate(dst,None)
        #ret,thresh = cv2.threshold(gray,50,255,0)
        #contours,hierarchy = cv2.findContours(thresh, 1, 2)
        #print("Number of contours detected:", len(contours))  

        # Threshold for an optimal value, it may vary depending on the image.
        print("Print number of corner %d", inference_img.shape)
        inference_img[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow("Inference", inference_img)

    """
    filename = 'chessboard.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
    
    img = cv2.imread('shapes.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,50,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print("Number of contours detected:", len(contours))    

    for cnt in contours:
           x1,y1 = cnt[0][0]
           approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
           if len(approx) == 4:
              x, y, w, h = cv2.boundingRect(cnt)
              ratio = float(w)/h
              if ratio >= 0.9 and ratio <= 1.1:
                 img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
                 cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
              else:
                 cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                 img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

    cv2.imshow("Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    # ------------------------------------------------------

    cv2.imshow("Frame", frame)
    #cv2.imshow("Board img", boardImg)

    cv2.imshow("Check board", checkBoard)
    cv2.imshow("Adaptive Thres Method", adapted_feed)
    cv2.imshow("Bit Result", bitResult)

    

    if key & 0xFF == ord('q'):
        break


            

vid.release()
cv2.destroyAllWindows()
