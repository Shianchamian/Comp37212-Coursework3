import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================

# Trackbar update function
def update(val):
    numDisparities = cv2.getTrackbarPos('NumDisparities', 'Disparity') * 16
    blockSize = cv2.getTrackbarPos('BlockSize', 'Disparity') * 2 + 5

    if numDisparities < 16:
        numDisparities = 16
    if blockSize % 2 == 0:
        blockSize += 1

    disp_gray = getDisparityMap(imgL, imgR, numDisparities, blockSize)
    disp_gray_norm = np.interp(disp_gray, (disp_gray.min(), disp_gray.max()), (0.0, 1.0))

    disp_edge = getDisparityMap(edgeL, edgeR, numDisparities, blockSize)
    disp_edge_norm = np.interp(disp_edge, (disp_edge.min(), disp_edge.max()), (0.0, 255.0)).astype(np.uint8)

    combined = np.hstack((disp_gray_norm, disp_edge_norm))
    cv2.imshow('Disparity', combined)




# ================================================
#
def plot(disparity, baseline, f, doffs):
    h, w = disparity.shape
    Xs = []
    Ys = []
    Zs = []

    for y in range(0, h, 5):  # Sample every 5 rows to reduce computation
        for x in range(0, w, 5):  # Sample every 5 columns
            d = disparity[y, x]
            if d > 0.0:
                Z = (baseline * f) / (d + doffs)
                if 1000 < Z < 8000:  # Filter out points outside valid depth range (in mm)
                    X = (x - w / 2) * Z / f  # Convert image x-coordinate to real-world X
                    Y = (y - h / 2) * Z / f  # Convert image y-coordinate to real-world Y
                    Xs.append(X)
                    Ys.append(Y)
                    Zs.append(Z)



    fig = plt.figure(figsize=(15, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(Xs, Ys, Zs, s=0.5)
    ax1.set_title("3D view")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Top view (X-Z)
    ax2 = fig.add_subplot(132)
    ax2.scatter(Xs, Zs, s=0.5)
    ax2.set_title("Top View (X-Z)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    # Side view (Y-Z)
    ax3 = fig.add_subplot(133)
    ax3.scatter(Ys, Zs, s=0.5)
    ax3.set_title("Side View (Y-Z)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

    plt.tight_layout()
    plt.savefig("views.png")
    plt.show()



# ================================================
#
if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    edgeL = cv2.Canny(imgL, 50, 150)
    edgeR = cv2.Canny(imgR, 50, 150)


    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('NumDisparities', 'Disparity', 4, 10, update)
    cv2.createTrackbar('BlockSize', 'Disparity', 3, 10, update)

    update(None)

 

    numDisparities = cv2.getTrackbarPos('NumDisparities', 'Disparity') * 16
    blockSize = cv2.getTrackbarPos('BlockSize', 'Disparity') * 2 + 5
    if numDisparities < 16:
        numDisparities = 16
    if blockSize % 2 == 0:
        blockSize += 1
        
    disparity = getDisparityMap(edgeL, edgeR, numDisparities, blockSize)
    
    # plot(disparity, baseline=174.019, f=5806.559, doffs=114.291)
    plot(disparity, baseline=174.019, f=1451.64, doffs=28.57275)



    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:  
            break




    cv2.destroyAllWindows()
