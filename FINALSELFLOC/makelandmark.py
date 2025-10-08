import cv2
import numpy as np

markerID = 2 # Change as needed

# Constants
ppi = 72
inch2mm = 25.4
ppmm = ppi / inch2mm

a4width = 210.0
a4height = 297.0

width = int(np.round(a4width * ppmm))
height = int(np.round(a4height * ppmm))

markerPhysicalSize = 150
markerSize = int(np.round(markerPhysicalSize * ppmm))

# White A4 image
landmarkImage = np.ones((height, width), dtype=np.uint8) * 255

# ArUco dictionary
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate marker (new API)
markerImage = np.zeros((markerSize, markerSize), dtype=np.uint8)
cv2.aruco.generateImageMarker(arucoDict, markerID, markerSize, markerImage, 1)

# Center marker
startWidth = int(np.round((width - markerSize) / 2))
startHeight = int(np.round((height - markerSize) / 2))
landmarkImage[startHeight:startHeight+markerSize, startWidth:startWidth+markerSize] = markerImage

# Add marker ID text
cv2.putText(landmarkImage, str(markerID),
            (startWidth, startHeight - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)

# Save and show
cv2.imwrite(f"landmark{markerID}.png", landmarkImage)
cv2.imshow("Landmark", landmarkImage)
cv2.waitKey()
cv2.destroyAllWindows()
