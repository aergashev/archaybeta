import cv2

def detect_face(frame, detector):
    faces = detector(frame, 0)
    return faces

def get_gender_and_age(face_roi, gender_model, age_model, gender_categories, age_categories):
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), False, False)

    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = gender_categories[gender_preds[0].argmax()]

    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = age_categories[age_preds[0].argmax()]

    return gender, age
