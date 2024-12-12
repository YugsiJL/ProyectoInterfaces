import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import mediapipe as mp
from PIL import Image

# Función para calcular el EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Variables globales
running = False
paused = False
closed_time = 0
counter = 0
start_time = None
cap = None

# Función para iniciar el análisis
def start_analysis():
    global running, paused, start_time, closed_time, counter, cap
    running = True
    paused = False
    closed_time = 0
    counter = 0
    start_time = time.time()
    
    # Iniciar captura de video
    cap = cv2.VideoCapture(0)
    
    def run_analysis():
        global running, paused, closed_time, counter, cap
        
        # Configuración de MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        EAR_THRESHOLD = 0.2
        CONSECUTIVE_FRAMES = 3
        TIME_LIMIT = 60

        frame = None

        while running:
            if paused:
                if frame is not None:  # Verificar si frame tiene un valor
                    black_frame = np.zeros_like(frame)
                    cv2.imshow("Sistema PERCLOS", black_frame)
                cv2.waitKey(1)
                continue

            ret, frame = cap.read()
            if not ret:
                update_console("No se pudo acceder a la cámara.")
                running = False
                cv2.destroyAllWindows()
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extraer puntos de los ojos (MediaPipe índices)
                    left_eye_indices = [362, 385, 387, 263, 373, 380]
                    right_eye_indices = [33, 160, 158, 133, 153, 144]
                    
                    left_eye = [(face_landmarks.landmark[i].x * frame.shape[1], 
                                 face_landmarks.landmark[i].y * frame.shape[0]) 
                                for i in left_eye_indices]
                    
                    right_eye = [(face_landmarks.landmark[i].x * frame.shape[1], 
                                  face_landmarks.landmark[i].y * frame.shape[0]) 
                                 for i in right_eye_indices]
                    
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Dibujar ojos en el frame
                    for point in left_eye + right_eye:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

                    if avg_ear < EAR_THRESHOLD:
                        counter += 1
                    else:
                        if counter >= CONSECUTIVE_FRAMES:
                            closed_time += counter / 30.0
                        counter = 0
                    
                    cv2.putText(frame, f"Presiona la tecla (q) para finalizar el analisis", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Tiempo ojos cerrados: {closed_time:.2f} s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Sistema PERCLOS", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

            if time.time() - start_time > TIME_LIMIT:
                running = False
                break

        percentage_closed = (closed_time / TIME_LIMIT) * 100
        update_console(f"Tiempo total ojos cerrados: {closed_time:.2f} segundos.")
        update_console(f"Porcentaje tiempo ojos cerrados: {percentage_closed:.2f}%.")
        if percentage_closed >= 30:
            update_console("Conclusión: Se detecta somnolencia.")
        else:
            update_console("Conclusión: No se detecta somnolencia.")

        cap.release()
        cv2.destroyAllWindows()
    
    threading.Thread(target=run_analysis, daemon=True).start()

# Función para mostrar el video en un label
def show_frame():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tk = tk.PhotoImage(image=Image.fromarray(cv2image))
    panel.imgtk = img_tk
    panel.config(image=img_tk)
    panel.after(10, show_frame)

# Función para capturar foto o grabar video
def capture_or_record():
    global is_recording, cap

    if not is_recording:
        # Capturar una foto
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("foto.jpg", frame)
            print("Foto capturada exitosamente.")
        else:
            print("Error al capturar la foto.")

        # Iniciar grabación de video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        is_recording = True

    else:
        # Detener grabación de video
        out.release()
        is_recording = False
        print("Grabación detenida.")

# Función para pausar el análisis
def pause_analysis():
    global paused
    paused = not paused
    update_console("Análisis pausado." if paused else "Análisis reanudado.")

# Función para detener el análisis
def stop_analysis():
    global running, cap
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    update_console("Análisis detenido.")

# Actualizar consola
def update_console(message):
    console.insert(tk.END, f"{message}\n")
    console.see(tk.END)

# Crear interfaz gráfica
root = tk.Tk()
root.title("Sistema PERCLOS")

# Diseño de GUI
panel = tk.Label(root, bg="black")
panel.pack(fill="both", expand="yes")

cap = cv2.VideoCapture(0)
thread = threading.Thread(target=show_frame)
thread.start()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Iniciar Análisis", command=start_analysis, bg="green", fg="white", width=15)
start_button.grid(row=0, column=0, padx=10)

capture_button = tk.Button(button_frame, text="", command=capture_or_record, bg="gray")
capture_button.grid(row=0, column=1, padx=10)

pause_button = tk.Button(button_frame, text="Pausar Análisis", command=pause_analysis, bg="yellow", fg="black", width=15)
pause_button.grid(row=0, column=1, padx=10)

console = scrolledtext.ScrolledText(root, width=60, height=10, state='normal')
console.pack(pady=10)

# Ejecutar GUI
root.mainloop()