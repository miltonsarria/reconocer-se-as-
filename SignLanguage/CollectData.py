import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2)
                             )
    
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose,face,lh,rh])


DATA_PATH = os.path.join('MP_Data')
#Actions
actions = np.array(['hola','gracias','iloveyou'])
#50 videos worth of data
no_sequences = 50
#30 frames
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)

#Mediapipe Model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    #Loop through Actions
    for action in actions:
        #Loop through Videos
        for sequence in range(no_sequences):
            #Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                
        
                #Read Feed
                ret, frame = cap.read()

                #Make detections
                image,results = mediapipe_detection(frame,holistic)

                #Draw Styled Landmarks
                draw_styled_landmarks(image,results)
                    # Dibujar landmarks
                #draw_styled_landmarks(image, results, mp_holistic, mp_drawing)

                image = cv2.flip(image, 1)

                cv2.putText(
                        image, f'Recolectando frames para la accion {action} | Video numero {sequence}',
                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                    )

                
                # Tiempo de espera para preparar
                if frame_num == 0:
                        cv2.putText(
                            image, "COMENZANDO RECOLECCIÓN", (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )
                        cv2.imshow('OpenCV', image)
                        cv2.waitKey(2000)
                else:
                        # Mostrar en pantalla
                        cv2.imshow('OpenCV', image)

                #NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH,action,str(sequence), str(frame_num))
                np.save(npy_path,keypoints)
                
                #mostrar
                cv2.imshow('OpenCV', image)

                #salida
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows() 
    
    
'''   
    
        # Acceso/configuración al modulo de mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


        # Ciclo para recorrer todos los recursos (letras y/o expresiones)
        for resource in resources:
            # Ciclo para recorrer todas las secuencias a recolectar
            for sequence in range(N_SEQ):
                # Ciclo para recoletar secuencias
                for n_frame in range(SEQ_LEN):

                    # Lectura via camara
                    _, frame = captura.read()

                    # Realizar detecciones
                    image, results = mediapipe_detection(frame, holistic)

                    # Dibujar landmarks
                    draw_styled_landmarks(image, results, mp_holistic, mp_drawing)

                    image = cv2.flip(image, 1)

                    cv2.putText(
                        image, f'Recolectando frames para la accion {resource} | Video numero {sequence}',
                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                    )

                    # Tiempo de espera para preparar
                    if n_frame == 0:
                        cv2.putText(
                            image, "COMENZANDO RECOLECCIÓN", (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )
                        cv2.imshow('OpenCV', image)
                        cv2.waitKey(2000)
                    else:
                        # Mostrar en pantalla
                        cv2.imshow('OpenCV', image)

                    # Exportar puntos claves
                    _, keypoints = extract_keypoints(results)
                    npy_path = DATA_PATH.joinpath(f'{resource}/{sequence}/{n_frame}')
                    np.save(npy_path, keypoints)


                    # Salida
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        captura.release()
        cv2.destroyAllWindows()
'''