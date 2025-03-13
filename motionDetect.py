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

#White Black threshhold (Lower = more sensitive to noise but captures smaller movements, Higher = less noise but captures less movements)
WBTHRESH = 135

#Threshhold for detecting a contour (Lower = more and smaller contours detected, Higher = less and larger contours detected)
CONTOURTHRESH = 100

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
        _, final = cv2.threshold(final, WBTHRESH, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > CONTOURTHRESH:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame1, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # store top-left coordinates
                coord_message = f"{x},{y}\n"
                print(f"Motion_coords: {coord_message.strip()}")

        cv2.imshow("Motion Detection", frame1)
        cv2.imshow("Foreground Mask", final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()