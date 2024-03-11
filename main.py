import cv2
import dlib
import torch
from l2cs import Pipeline, render
from utils import detect_face, get_gender_and_age

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r".\shape_predictor_68_face_landmarks.dat")

gender_model = cv2.dnn.readNet(r".\gender_deploy.prototxt", r".\gender_net.caffemodel")
age_model = cv2.dnn.readNet(r".\age_deploy.prototxt", r".\age_net.caffemodel")

gender_categories = ['Male', 'Female']
age_categories = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gaze_pipeline = Pipeline(
    weights='models\Gaze360\L2CSNet_gaze360.pkl',  # e.g., 'models/L2CSNet_gaze360.pkl'
    arch='ResNet50',
    device=torch.device('cpu')  # or 'gpu'
)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = detect_face(frame, detector)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        gender, age = get_gender_and_age(face_roi, gender_model, age_model, gender_categories, age_categories)

        # Process frame and visualize gaze
        results = gaze_pipeline.step(frame)
        frame = render(frame, results)

        cv2.putText(frame, f"Gender: {gender}, Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Gender, Age, and Gaze Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
