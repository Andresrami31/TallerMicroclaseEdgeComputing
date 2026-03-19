from ultralytics import YOLO
import sys

# Cargar modelo — no modificar
modelo = YOLO("yolov8n.pt")

# Analizar imagen — no modificar
imagen = sys.argv[1]
resultados = modelo(imagen)

lineas = []
etiquetas = []

for resultado in resultados:
    etiquetas = resultado.boxes.cls.cpu().numpy().tolist()
    etiquetas = [modelo.names[int(etiqueta)] for etiqueta in etiquetas]
    lineas.append(f"Imagen elegida: {imagen}\n")
    lineas.append(f"Etiquetas detectadas: {etiquetas}\n")

# Guardar — no modificar
with open("resultados.txt", "w", encoding="utf-8") as f:
    f.writelines(lineas)

print("✅ Resultados guardados en resultados.txt")