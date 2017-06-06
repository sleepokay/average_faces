from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
import argparse
import traceback
import shutil

# initialize dlib's face detector (HOG-based)
# recreate the facial landmark predictor from model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_landmarks(img_path):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=150)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # detect facial bounding boxes in the grayscale image
    boxes = detector(gray, 0)

    # we only want to work with pictures with one face
    if len(boxes) != 1:
    	return

    box = boxes[0]

    shape = predictor(gray, box) # identify facial features
    landmarks = face_utils.shape_to_np(shape) # shape to numpy array

    return landmarks


def write_landmarks(output_path, landmarks):
    f = open(output_path, 'w')

    for (x, y) in landmarks:
    	# cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        f.write(str(x) + " " + str(y) + "\n")

    f.close()

 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input image directory")
    ap.add_argument("-o", "--output", required=True, help="output facial landmarks directory")
    args = vars(ap.parse_args())


    print(os.listdir(args["input"]))

    try:
        for file in os.listdir(args["input"]):
            print(file)
            if file.endswith(".jpg"):
                img_path = os.path.join(args["input"], file)
                landmarks = detect_landmarks(img_path)
                if landmarks != None:
                    print(args["output"] +  "/" + file[0:2] + "/");
                    if not os.path.exists(args["output"] + "/" + file[0:2] + "/"):
                        os.makedirs(args["output"] +  "/" + file[0:2] + "/")
                    shutil.copy2(img_path, args["output"] +  "/" + file[0:2] + "/")
                    write_landmarks(args["output"] + "/" + file[0:2] + "/" + file + ".txt", landmarks)

    except Exception as e:
        print("Unexpected error")
        print(e)
        traceback.print_exc()