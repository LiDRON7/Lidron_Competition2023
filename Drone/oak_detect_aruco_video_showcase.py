# first, import all necessary modules
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np
from cv2 import cuda

# Necessary imports for ArUco detection
import argparse
import imutils
import time
import sys


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_6X6_100",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())


# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50, 
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50, # default
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
	sys.exit(0)
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
# arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_50"])
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])# dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
# parameters =  cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")


# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

# Next, we want a neural network that will produce the detections
# detection_nn = pipeline.createMobileNetDetectionNetwork()

# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
# detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))

# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
# detection_nn.setConfidenceThreshold(0.5)

# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
# cam_rgb.preview.link(detection_nn.input)


# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
# detection_nn.out.link(xout_nn.input)


# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    detections = []

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    # Main host-side application loop
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections

        # if frame is not None:
        #     for detection in detections:
        #         # for each bounding box, we first normalize it to match the frame size
        #         bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        #         # and then draw a rectangle on the frame to show the actual result
        #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        #     # After all the drawing is finished, we show the frame on the screen
        #     cv2.imshow("preview", frame)


        if frame is not None:
            frame = imutils.resize(frame, width=1000)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
            # (corners, ids, rejected) = detector.detectMarkers(frame)

            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:

		        # flatten the ArUco IDs list
                ids = ids.flatten()
                # print("Detected at least one ArUco marker")
		        # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
			        # in top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
			        # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
			

                    # draw the bounding box of the ArUCo detection for friendly units
                    if (markerID == 12):
                        cv2.line(frame, topLeft, topRight, (0, 255, 0), 8)
                        cv2.line(frame, topRight, bottomRight, (0, 255, 0), 8)
                        cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 8)
                        cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 8)
                        cv2.putText(frame, f"{markerID}: F R I E N D", (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    else:
                        cv2.line(frame, topLeft, topRight, (0, 0, 255), 8)
                        cv2.line(frame, topRight, bottomRight, (0, 0, 255), 8)
                        cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 8)
                        cv2.line(frame, bottomLeft, topLeft, (0, 0, 255), 8)
                        cv2.putText(frame, f"{markerID}: E N E M Y", (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


			        # compute and draw the center (x, y)-coordinates of the
			        # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
			        # draw the ArUco marker ID on the frame
                    # cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			
	        # show the output frame
            cv2.imshow("Frame", frame)
            
        key = cv2.waitKey(1) & 0xFF
	    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break




        # # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        # if cv2.waitKey(1) == ord('q'):
        #     break
