# -*- coding: utf-8 -*-
import dlib
import cv2
import numpy as np

# ==============================================================================
#   1.landmarks format conversion functions
#       Input：landmarks in dlib format
#       Output：landmarks in numpy format
# ==============================================================================


def landmarks_to_np(landmarks, dtype="int"):
    # Get the number of landmarks
    num = landmarks.num_parts

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

# ==============================================================================
#   **************************Main function entry***********************************
# ==============================================================================


# Face keypoint training data path
predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()  # Face detector
# Face keypoint detector predictor
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

# Initialising a time series queue
queue = np.zeros(30, dtype=int)
queue = queue.tolist()

while (cap.isOpened()):
    # Reading video frames
    _, img = cap.read()

    # Convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detection
    rects = detector(gray, 1)

    # Operate on each detected face
    for i, rect in enumerate(rects):
        # Get coordinates

        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # Drawing borders, adding text labels
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Testing landmarks
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        # Marked with landmarks
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # Calculating the Euclidean distance
        d1 = np.linalg.norm(landmarks[37]-landmarks[41])
        d2 = np.linalg.norm(landmarks[38]-landmarks[40])
        d3 = np.linalg.norm(landmarks[43]-landmarks[47])
        d4 = np.linalg.norm(landmarks[44]-landmarks[46])
        d_mean = (d1+d2+d3+d4)/4
        d5 = np.linalg.norm(landmarks[36]-landmarks[39])
        d6 = np.linalg.norm(landmarks[42]-landmarks[45])
        d_reference = (d5+d6)/2
        d_judge = d_mean/d_reference
        print(d_judge)

        # Eye open/closed flag: based on the threshold to determine whether the eyes are closed, closed flag=1, open flag=0 (threshold adjustable)
        flag = int(d_judge < 0.25)

        # flag into the team
        queue = queue[1:len(queue)] + [flag]

        # Determination of fatigue: based on whether more than half of the elements in the time series are below the threshold
        if sum(queue) > len(queue)/2:
            cv2.putText(img, "WARNING !", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "SAFE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Show results
    cv2.imshow("Result", img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:   # Press "Esc" to exit
        break

cap.release()
cv2.destroyAllWindows()
