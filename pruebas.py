import cv2

# Iniciar la captura de video
cap = cv2.VideoCapture(0)  # 0 es el índice de la cámara

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

while True:
    # Capturar frame a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se puede recibir el frame.")
        break

    # Mostrar el frame
    cv2.imshow('Cámara', frame)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
