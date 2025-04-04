import numpy as np
import cv2
import sys

def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

def update(val):
    numDisparities = cv2.getTrackbarPos('NumDisparities', 'Focus') * 16
    blockSize = cv2.getTrackbarPos('BlockSize', 'Focus') * 2 + 5
    # k = cv2.getTrackbarPos('k', 'Focus') + 1
    raw_k = cv2.getTrackbarPos('k', 'Focus')  # raw ∈ [0, 50]
    k = raw_k / 10.0 + 0.1  # k ∈ [0.1, 5.1]

    threshold = 200

    disp = getDisparityMap(imgL_gray, imgR_gray, numDisparities, blockSize)

    # Compute depth from disparity
    depth = 1.0 / (disp + k)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Thresholding (smaller = closer = foreground)
    _, mask = cv2.threshold(depth, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)
    mask_inv = cv2.bitwise_not(mask)

    # Extract foreground using the mask
    fg = cv2.bitwise_and(imgL, imgL, mask=mask)

    # Blur the background
    blurred = cv2.GaussianBlur(imgL, (15, 15), 0)
    bg = cv2.bitwise_and(blurred, blurred, mask=mask_inv)

    combined = cv2.add(fg, bg)

    # Display the final result
    cv2.imshow("Selective Focus", combined)


if __name__ == '__main__':
    imgL = cv2.imread('girlL.png')
    imgR = cv2.imread('girlR.png')
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if imgL is None or imgR is None:
        print("Error loading images")
        sys.exit()

    cv2.namedWindow('Focus', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('NumDisparities', 'Focus', 4, 10, update)
    cv2.createTrackbar('BlockSize', 'Focus', 10, 10, update)
    cv2.createTrackbar('k', 'Focus', 10, 50, update)

    update(None)

    while True:
        key = cv2.waitKey(1)
        if key == 27 or key == ord(' '):  # Exit on ESC or SPACE
            break

    cv2.destroyAllWindows()
