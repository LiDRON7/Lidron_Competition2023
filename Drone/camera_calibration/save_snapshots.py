"""
Saves a series of snapshots with the current camera as snapshot_<width>_<height>_<nnn>.jpg

Arguments:
    --f <output folder>     default: current folder
    --n <file name>         default: snapshot
    --w <width px>          default: none
    --h <height px>         default: none

Buttons:
    q           - quit
    space bar   - save the snapshot
    
  
"""

import cv2
import time
import sys
import argparse
import os
import depthai
import numpy as np

__author__ = "Tiziano Fiorenzani"
__date__ = "01/06/2018"


def save_snaps(width=1080, height=720, name="snapshot", folder=".", raspi=False):

    if raspi:
        os.system('sudo modprobe bcm2835-v4l2')
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(1080, 720)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)

    cam_rgb_preview = pipeline.createXLinkOut()
    cam_rgb_preview.setStreamName("rgb_preview")
    cam_rgb.preview.link(cam_rgb_preview.input)

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
            # ----------- CREATE THE FOLDER -----------------
            folder = os.path.dirname(folder)
            try:
                os.stat(folder)
            except:
                os.mkdir(folder)
    except:
        pass

    nSnap   = 0
    

    fileName    = "%s/%s" %(folder, name)
    with depthai.Device(pipeline) as device:
        while True:
            frame = device.getOutputQueue("rgb_preview").tryGet()
            if frame is None:
                continue
            frame_data = frame.getData().reshape(3, frame.getHeight(), frame.getWidth()).transpose(1, 2, 0).astype(np.uint8)
            cv2.imshow('Frame', frame_data)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                print("Saving image ", nSnap)
                cv2.imwrite("%s%d.jpg"%(fileName, nSnap), frame_data)
                nSnap += 1

    cv2.destroyAllWindows()




def main():
    # ---- DEFAULT VALUES ---
    SAVE_FOLDER = "."
    FILE_NAME = "snapshot"
    FRAME_WIDTH = 0
    FRAME_HEIGHT = 0

    # ----------- PARSE THE INPUTS -----------------
    parser = argparse.ArgumentParser(
        description="Saves snapshot from the camera. \n q to quit \n spacebar to save the snapshot")
    parser.add_argument("--folder", default=SAVE_FOLDER, help="Path to the save folder (default: current)")
    parser.add_argument("--name", default=FILE_NAME, help="Picture file name (default: snapshot)")
    parser.add_argument("--dwidth", default=FRAME_WIDTH, type=int, help="<width> px (default the camera output)")
    parser.add_argument("--dheight", default=FRAME_HEIGHT, type=int, help="<height> px (default the camera output)")
    parser.add_argument("--raspi", default=False, type=bool, help="<bool> True if using a raspberry Pi")
    args = parser.parse_args()

    SAVE_FOLDER = args.folder
    FILE_NAME = args.name
    FRAME_WIDTH = args.dwidth
    FRAME_HEIGHT = args.dheight


    save_snaps(width=args.dwidth, height=args.dheight, name=args.name, folder=args.folder, raspi=args.raspi)

    print("Files saved")

if __name__ == "__main__":
    main()



