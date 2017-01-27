import time
import cv2
import numpy as np
from picamera import PiCamera, PiCameraCircularIO
from SimpleCV import Image, ColorSpace

class Camera():

    def __init__(self, size=1, frameRate=40, hflip=False, vflip=False):
        """A wrapper class for the Raspberry Pi camera using the picamera
        python library. The size parameter sets the camera resolution to
        size * (64, 48)."""
        self.active = False
        try:
            if type(size) is not int:
                raise TypeError
            elif 1 <= size and size <= 51:
                self.size = size
                self.hRes = size * 64
                self.vRes = size * 48
            else:
                raise ValueError
        except TypeError:
            raise TypeError("Size must be an integer")
        except ValueError:
            raise ValueError("Size must be in range 1 to 51")
        self.picam = PiCamera()
        self.picam.resolution = (self.hRes, self.vRes)
        self.picam.framerate = frameRate
        self.picam.hflip = hflip
        self.picam.vflip = vflip
        time.sleep(1)
        self.stream = PiCameraCircularIO(self.picam, seconds=1)
        self.start()

    def close(self):
        """Stops the running thread and closes the PiCamera instance."""
        self.stop()
        self.picam.close()

    def doWitheBalance(self, awbFilename='awb_gains.txt', mode='auto'):
        """A method that performs white balance calibration, sets the PiCamera
        awb_gains to fixed values and write these values in a file. For best
        results, put a white objet in the camera field of view (a sheet of paper
        ) during the calibration process."""
        ##  Set AWB mode for calibration
        self.picam.awb_mode = mode
        print 'Calibrating white balance gains...'
        time.sleep(1)
        ##  Read AWB gains
        gains = self.picam.awb_gains
        ##  Set AWB mode to off (manual)
        self.picam.awb_mode = 'off'
        ##  Set AWB gains to remain constant
        self.picam.awb_gains = gains

        ##  Write AWB gains to file
        gRed = float(gains[0])
        gBlue = float(gains[1])
        f = open(awbFilename, 'w')
        f.flush()
        f.write(str(gRed) + ', ' + str(gBlue))
        f.close()
        print 'AWB gains set to:', gRed, gBlue
        print 'AWB gains written to ' + awbFilename

    def getOpenCVImage(self):
        """Grabs a frame from the camera and returns an OpenCV image array."""
        img = np.empty((self.vRes * self.hRes * 3), dtype=np.uint8)
        self.picam.capture(img, 'bgr', use_video_port=True)
        img = img.reshape((self.vRes, self.hRes, 3))
        return img

    def getSimpleCVImage(self):
        """Grabs a frame from the camera and returns a SimpleCV image object."""
        img = np.empty((self.vRes * self.hRes * 3), dtype=np.uint8)
        self.picam.capture(img, 'bgr', use_video_port=True)
        img = img.reshape((self.vRes, self.hRes, 3))
        img = Image(img, colorSpace=ColorSpace.RGB)
        img = img.rotate90()
        img = img.flipVertical()
        return img

    def kill(self):
        """Same as .close() method."""
        self.close()

    def readWhiteBalance(self, awbFilename='awb_gains.txt'):
        """Reads white balance gains from a file created using the
        .doWitheBalance() method and fixes the PiCamera awb_gains parameter
        to these values."""
        ##  Read AWB gains from file
        f = open(awbFilename, 'r')
        line = f.readline()
        f.close()
        gRed, gBlue = [float(g) for g in line.split(', ')]
        ##  Set AWB mode to off (manual)
        self.picam.awb_mode = 'off'
        ##  Set AWB gains to remain constant
        self.picam.awb_gains = gRed, gBlue
        print 'AWB gains set to:', gRed, gBlue

    def start(self):
        """Starts continuous recording of the camera into a PicameraCircularIO
        buffer."""
        if not self.active:
            self.active = True
            self.picam.start_recording(self.stream, format='h264',
                                       resize=(self.hRes, self.vRes))

    def startPreview(self):
        """Starts the preview of the PiCamera. Works only on the display
        connected directly on the Raspberry Pi."""
        self.picam.start_preview()

    def stop(self):
        """Stops the camera continuous recording and stops the preview if
        active."""
        self.active = False
        self.picam.stop_recording()
        self.stopPreview()

    def stopPreview(self):
        """Stops the PiCamera preview if active."""
        self.picam.stop_preview()
