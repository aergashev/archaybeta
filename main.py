import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r".\shape_predictor_68_face_landmarks.dat")

gender_model = cv2.dnn.readNet(r".\deploy_gender.prototxt", r".\gender_net.caffemodel")
age_model = cv2.dnn.readNet(r".\deploy_age.prototxt", r".\age_net.caffemodel")

gender_categories = ['Male', 'Female']
age_categories = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = detector(frame, 0)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), False, False)

        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = gender_categories[gender_preds[0].argmax()]

        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = age_categories[age_preds[0].argmax()]

        cv2.putText(frame, f"Gender: {gender}, Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Gender and Age Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
