import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Tipo de clasificación
clases = {0: "Extra", 1: "Cat_I", 2: "Cat_II", 3: "Cat_III"}

# Cargando Modelo
modelo = load_model(r'C:\Users\UPN\Desktop\pro\modelo_limones.h5')
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # webcam

# Establecer resolución de la cámara a 1:1 (NO ME FUNCIONA)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)  # Ancho
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Alto

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error.")
        break

    # Obtener dimensiones del frame
    height, width = frame.shape[:2]
    
    # Convertir en 1:1
    size = min(height, width)
    x_start = (width - size) // 2
    y_start = (height - size) // 2

    # Recortar la imagen para que sea cuadrada
    imagen_cuadrada = frame[y_start:y_start + size, x_start:x_start + size]
    cv.imshow("square", imagen_cuadrada)  # Comentar luego de pruebas

    # Redimensionar
    imagen_resized = cv.resize(imagen_cuadrada, (180, 180)) 
    # Normalizando
    imagen_normalizada = imagen_resized / 255.0
    # Dimensión extra para el batch size
    entrada = np.expand_dims(imagen_normalizada, axis=0)

    # Predicción
    prediccion = modelo.predict(entrada)
    clase = np.argmax(prediccion)
    indice_max = np.argmax(prediccion)

    # Mostrar el porcentaje de predicción y clase predicha en la ventana
    cv.putText(frame, f"Porcentaje: {prediccion[0][indice_max] * 100:.2f}%", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, f"Clase: {clases[clase]}", (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Con fondo BLANCO
    # Convertir a espacio de color HSV(Tono,Saturacion,valor)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # COLOR DE LOS LIMONES
    lower_yellow = np.array([20, 100, 100])  # Min: Matiz, Saturación, Valor
    upper_yellow = np.array([30, 255, 255])  # Max: Matiz, Saturación, Valor
    lower_green = np.array([35, 100, 100])  
    upper_green = np.array([85, 255, 255]) 
    lower_brown = np.array([10, 100, 20])  
    upper_brown = np.array([20, 255, 200]) 
    # Crear máscaras para amarillo y verde
    mask_yellow = cv.inRange(hsv_frame, lower_yellow, upper_yellow)
    mask_green = cv.inRange(hsv_frame, lower_green, upper_green)
    mask_brown = cv.inRange(hsv_frame, lower_brown, upper_brown)
    # Encontrar contornos en la máscara
    contours_yellow, _ = cv.findContours(mask_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv.findContours(mask_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_brown, _ = cv.findContours(mask_brown, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # unir mascaras
    contours = contours_yellow + contours_green + contours_brown

    if contours:
        # Unir contornos
        all_contours = np.vstack(contours)

        # Calcular un único rectángulo que encierre todos los contornos
        x, y, w, h = cv.boundingRect(all_contours)

        # Dibujar un único rectángulo alrededor del limón
        if cv.contourArea(all_contours) > 200:  
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #video 
    cv.imshow('Predicción', frame)

    if cv.waitKey(1) & 0xFF == ord('a'):
        break