import cv2
# load some some pre-trained data front face from openCV Haar cascade algorithm
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an image to detect the face in
img = cv2.imread('rdj.jpg')
# img = cv2.imread('jamesBond.png')

# convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# DETECT FACES
face_cordinations = trained_face_data.detectMultiScale(grayscale_img)


# draw rectangle around the face and multiple faces
for (x, y, w, h) in face_cordinations:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# image show
cv2.imshow('Face Detector', img)
# waitkey is import to to hold the image screen and press any key to close the window
cv2.waitKey()
