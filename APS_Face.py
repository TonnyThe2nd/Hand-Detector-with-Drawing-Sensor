import cv2
import mediapipe as mp
import math
import numpy

video = cv2.VideoCapture(0)

faceMash = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
face_rec_config = faceMash.FaceMesh(
    max_num_faces = 1,
    min_detection_confidence = 0.8,
    min_tracking_confidence = 0.7,
    static_image_mode = False
)


while True:

    ret, cam = video.read()
    cam = cv2.flip(cam, 1)
    frame_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    detect_face = face_rec_config.process(frame_rgb)

    if detect_face.multi_face_landmarks:
        for face in detect_face.multi_face_landmarks:
            mp_draw.draw_landmarks(image=cam,
                    landmark_list=face,
                    connections=faceMash.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style())
    cv2.imshow('Camera Facial', cam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
