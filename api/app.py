# api/app.py
import io
import numpy as np
import cv2
import librosa
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Reutilizados de nuestro nuevo módulo compartido para no duplicar código
from modelos_audio import CNNAudioUrbano, cargar_modelo_cnn, clasificar_audio  

app = Flask(__name__)

# ─── Carga única de modelos al iniciar el proceso ───────────────────────
MODELO_YOLO = YOLO("modelos/yolov8_best.pt")
MODELO_CNN = cargar_modelo_cnn("modelos/cnn_audio_urbansound.pt")

@app.route("/predict/video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "Se requiere un archivo de imagen en 'file'"}), 400
    
    file_bytes = np.frombuffer(request.files["file"].read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 400
        
    resultados = MODELO_YOLO(frame, verbose=False, conf=0.3)
    detecciones = []
    
    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detecciones.append({
                "clase": MODELO_YOLO.names[int(box.cls[0])],
                "confianza": float(box.conf[0]),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
            
    return jsonify({"total_objetos": len(detecciones), "detecciones": detecciones})

@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "Se requiere un archivo de audio en 'file'"}), 400
        
    audio_bytes = io.BytesIO(request.files["file"].read())
    audio_array, sr = librosa.load(audio_bytes, sr=22050, mono=True)
    
    clase, confianza = clasificar_audio(MODELO_CNN, audio_array, sr)
    
    return jsonify({"clase_audio": clase, "confianza": confianza})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)