import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import requests
import base64

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Initialize time variables
start_time = time.time()
interval = 5  # Time interval in seconds

image_count = 1

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        # Check if imgCrop is not empty
        if imgCropShape[0] != 0 and imgCropShape[1] != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("ImageCrop", imgCrop)

            # Convert the image to base64 for sending
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')


    cv2.imshow("image", img)

    # Check if the interval has elapsed
    if time.time() - start_time >= interval:
        # Save the detected hand image
        cv2.imwrite(f"./images/captured_image_{image_count}.jpg", imgCrop)
        print(f"Image {image_count} saved!")
        image_count += 1
        start_time = time.time()  # Reset the start time

    key = cv2.waitKey(1)

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
