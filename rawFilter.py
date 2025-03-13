import cv2
import numpy as np
import time

#Define cameras (These values will vary depending on how many cameras your computer has)
LOCALCAM = 0
WEBCAM = 4

#Set camera to use
CAM = LOCALCAM

#Set time offset between frame captures (Longer = slow motion, Shorter = fast motion)
OFFSET = 0

def main():
    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret1, frame1 = cap.read()
        time.sleep(OFFSET)
        ret2, frame2 = cap.read()

        if not ret1 and not ret2:
            print("Stream ended or failed.")
            break

        invert = cv2.bitwise_not(frame2)
        overlay = cv2.addWeighted(invert, 0.5, frame1, 0.5, 0)
        final = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Motion Detection", frame1)
        cv2.imshow("Foreground Mask", final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()