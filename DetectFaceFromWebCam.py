import cv2
# load some some pre-trained data front face from openCV Haar cascade algorithm
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# webcam = cv2.VideoCapture("newYork.mp4")


# capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    # reade current frame
    successful_frame_read, frame = webcam.read()
    if frame == None:
        print("Check Your WebCam Something went wrong")
        break  # if you remove this statement print above statement infinite

    else:
        # convert to grayscale
        grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # DETECT FACES
        face_cordinations = trained_face_data.detectMultiScale(grayscale_img)

        # draw rectangle around the face and multiple faces
        for (x, y, w, h) in face_cordinations:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imshow('Face Detector', frame)
        key = cv2.waitKey(1)

        # press Q key for quit
        if key == 81 or key == 113:
            break

# release the webcam
webcam.release()
