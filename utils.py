import cv2
import dlib

# Load the face detection model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r".\shape_predictor_68_face_landmarks.dat")

# Load the gender detection model
gender_net = cv2.dnn.readNetFromCaffe(r".\deploy_gender.prototxt", r".\gender_net.caffemodel")

# Load the age detection model
age_net = cv2.dnn.readNetFromCaffe(r".\deploy_age.prototxt", r".\age_net.caffemodel")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    return faces, gray

def get_gender_and_age(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Gender detection
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

    # Age detection
    age_net.setInput(blob)
    age_preds = age_net.forward()

    # Find the index with the maximum probability
    max_prob_index = max(range(len(age_preds[0])), key=lambda i: age_preds[0][i])
    
    # Calculate the estimated age
    age = max_prob_index

    return gender, age


