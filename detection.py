import cv2
import numpy as np
import serial
import time

# Initialize serial connection to Arduino
# arduino_port = '/dev/ttyUSB0'  # Change this based on your system (e.g., 'COM3' for Windows)
# baud_rate = 9600
# arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Allow Arduino time to initialize

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    while True:
        ret1, frame1 = cap.read()
        #time.sleep(0.05)
        ret2, frame2 = cap.read()

        if not ret1 and not ret2:
            print("Stream ended or failed.")
            break

        invert = cv2.bitwise_not(frame2)
        overlay = cv2.addWeighted(invert, 0.5, frame1, 0.5, 0)
        final = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, final = cv2.threshold(final, 135, 255, cv2.THRESH_BINARY)

        # gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # fg_mask = backSub.apply(gray)
        # kernel = np.ones((5, 5), np.uint8)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Send top-left coordinates to Arduino
            coord_message = f"{x},{y}\n"
            # arduino.write(coord_message.encode())
            print(f"Sent to Arduino: {coord_message.strip()}")

        cv2.imshow("Motion Detection", frame1)
        cv2.imshow("Foreground Mask", final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    # arduino.close()

if __name__ == "__main__":
    main()