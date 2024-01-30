from tensorflow.keras.models import load_model

model = load_model('/home/sinsagar/Downloads/garbage_classification.h5') 
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your pre-trained model
#model = load_model('Garbage Classification/garbage_classification.h5')

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def detect_garbage():
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break

        frame = cv2.resize(frame, (150, 150))
        preprocessed_frame = np.expand_dims(frame, axis=0)
        preprocessed_frame = preprocessed_frame / 255.0

        predictions = model.predict(preprocessed_frame)
        predicted_class_idx = np.argmax(predictions)
        predicted_class_label = labels[predicted_class_idx]

        frame = cv2.resize(frame, (640, 500))
        cv2.putText(frame, predicted_class_label, (300, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('video_capture', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

detect_garbage()

