import threading
import time
import cv2
import numpy as np

class LineTrackerBox():

    def __init__(self, cam): 
        """A class used to track a line using computer vision. Useful for line
        following or line maze solving. A Camera object instance with size
        parameter of 2 is passed to the class constructor. The .trackLine()
        method launches a thread that continuously takes images from the camera
        nd runs the .getIntBtmHPos() function to update the class intersection
        and btmHPos attributes. See description of .getIntBtmHPos() function for
        details on the meaning of these parameters."""
        self.active = False
        try:
            if cam.size == 2:
                self.cam = cam
            else:
                raise ValueError
        except ValueError:
            raise ValueError("Camera instance size attribute must equal 2")
        self.btmHPos = 0
        self.intersection = 0
        self.trackLine()

    def getBtmHPos(self):
        return self.btmHPos

    def getIntersection(self):
        return self.intersection

    def stop(self):
        """Stops the active thread running the .trackLine() method."""
        self.active = False
        time.sleep(.1)
        self.btmHPos = 0
        self.intersection = 0

    def _trackLine(self):
        """Private method that continuously takes images from the camera
            and runs the .getIntBtmHPos() method to update the class
            intersection and hPosBtm attributes."""
        while self.active:
            img = self.cam.getOpenCVImage()
            self.intersection, self.btmHPos = getIntBtmHPos(img)

    def trackLine(self):
        """Starts the .trackLine() method in a thread. Use the .stop()
            method to stop the thread."""
        if self.active:
            print "trackLine is already running"
        else:
            self.active = True
            th = threading.Thread(target=self._trackLine, args=[])
            th.start()

def findMaxAreaContour(img, minArea):
    """ Returns the contour with the largest area in img if its area is
        >= minArea. Returns None if no contour has area >= minArea."""

    ##  Extract contours
    contours, hier = cv2.findContours(img, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    ##  Check if contours found
    ret = None
    if len(contours) > 0:
        ##  Calculate areas
        areas = [cv2.contourArea(cnt) for cnt in contours]
        ##  Find largest area
        index = np.argmax(areas)
        ##  Check if >= minArea
        if areas[index] >= minArea:
            ret = contours[index]

    return ret

def getIntBtmHPos(img):
    """ Analyses an image to find the green tape lines. Returns:
        intersection: a number corresponding to the type of intersection:
            0:  Dead-end
            +1: There is a path to the left
            +2: There is a path forward
            +4: There is a path to the right
            8:  It is the end (large green rectangle)
            +1, +2 and +4 are added. For example, 3 means a path to the left
            and a path forward, 2 means a straight line, etc.
        hPosBtm: Horizontal position of the line at the bottom of the image.
                 Value bound within 1 (full left) to -1 (full right).
                 This value is useful as input for line following."""

    ## Convert image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## Crop to keep only bottom portion of image
    img = img[36:, :]

    ## Threshold green tape color
    lowGreen = np.array([45, 50, 50])
    highGreen = np.array([80, 255, 255])
    mask = cv2.inRange(img, lowGreen, highGreen)

    ## Dilate & erode to eliminate holes
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    ## Define Regions of Interest (ROIs)
    top = mask[:8, 8:120]
    btm = mask[52:, 8:120]
    left = mask[:, :8]
    right = mask[:, 120:]

    ## Find contours and keep largest area contour for each image
    ## Eliminate contour if smaller than minArea
    minArea = 10
    ## Top image
    cntTop = findMaxAreaContour(top, minArea)
    ## Btm image
    cntBtm = findMaxAreaContour(btm, minArea)
    ## Left image
    cntLeft = findMaxAreaContour(left, minArea)
    ## Right image
    cntRight = findMaxAreaContour(right, minArea)

    ## Calculate horizontal position of botom contour within btm
    if cntBtm is not None:
        ##  Calculate contour moments
        momentsBtm = cv2.moments(cntBtm)
        ##  Calculate centroid x position
        cxBtm = int(momentsBtm['m10']/momentsBtm['m00'])
        ##  Scale centroid x position from 1 (left) to -1 (right)
        hPosBtm = (btm.shape[1] / 2. - cxBtm) / (btm.shape[1] / 2.)
        ##  Calculate area
        areaBtm = cv2.contourArea(cntBtm)
    else:
        ##  If no contour at bottom set values to 0
        hPosBtm = 0
        areaBtm = 0

    ##  Evaluated intersection type
    intersection = 0


    if cntLeft is not None:
        ##  There is a path to the left
        intersection += 1
    if cntTop is not None:
        ##  There is a path fowrard
        intersection += 2
        areaTop = cv2.contourArea(cntTop)
    else:
        areaTop = 0
    if cntRight is not None:
        ##  There is a path to the right
        intersection += 4

    ##  Check if finish
    if areaTop >= 200 and areaBtm >= 200:
        intersection = 8

    return intersection, hPosBtm
