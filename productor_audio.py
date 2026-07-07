"""
Productor de Audio Kafka - Pipeline de Percepción Computacional
================================================================
Extrae el audio de trafico.mp4, lo procesa usando la CNN en
ventanas de 4 segundos, y envía los resultados a Kafka.
"""

import time
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import librosa
from moviepy import VideoFileClip
from kafka import KafkaProducer

# ─── Configuración ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELO_CNN = BASE_DIR / "modelos" / "cnn_audio_urbansound.pt"
VIDEO_PATH = BASE_DIR / "data" / "trafico.mp4"

KAFKA_BROKER = "localhost:9092"
TOPIC_AUDIO = "audio-stream"

# ─── Clases de UrbanSound8K ────────────────────────────────────────────────
CLASES_AUDIO = [
    "Aire acondicionado", "Bocina de auto", "Niños jugando", "Ladrido de perro",
    "Taladro", "Motor en marcha", "Disparo", "Martillo neumático", "Sirena", "Música callejera"
]

# ─── Definición de la CNN de Audio ──────────────────────────────────────────
class CNNAudioUrbano(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        self.bloque1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.bloque2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.bloque3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.clasificador = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16 * 21, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_clases))

    def forward(self, x):
        return self.clasificador(self.bloque3(self.bloque2(self.bloque1(x))))

def cargar_modelo_cnn(ruta):
    modelo = CNNAudioUrbano(num_clases=10)
    state_dict = torch.load(str(ruta), map_location="cpu", weights_only=False)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo

def clasificar_audio(modelo_cnn, audio_array, sr=22050):
    try:
        audio_mono = np.mean(audio_array, axis=1) if len(audio_array.shape) > 1 else audio_array
        audio_mono = audio_mono.astype(np.float32)
        target_len = sr * 4
        if len(audio_mono) < target_len:
            audio_mono = np.pad(audio_mono, (0, target_len - len(audio_mono)))
        else:
            audio_mono = audio_mono[:target_len]
        
        mel = librosa.feature.melspectrogram(y=audio_mono, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] > 173: mel_db = mel_db[:, :173]
        elif mel_db.shape[1] < 173: mel_db = np.pad(mel_db, ((0, 0), (0, 173 - mel_db.shape[1])))
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            salida = modelo_cnn(tensor)
            probabilidades = torch.softmax(salida, dim=1)
            confianza, clase_idx = torch.max(probabilidades, dim=1)
            
        return CLASES_AUDIO[clase_idx.item()], confianza.item()
    except Exception as e:
        return "Sin clasificar", 0.0

def procesar_audio():
    # Conectar a Kafka
    try:
        productor = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print(f"[OK] Conectado a Kafka: {KAFKA_BROKER}")
    except Exception as e:
        print(f"[ERROR] No se pudo conectar a Kafka: {e}")
        sys.exit(1)

    print(f"[INFO] Cargando modelo CNN desde: {MODELO_CNN}")
    modelo_cnn = cargar_modelo_cnn(MODELO_CNN)

    print(f"[INFO] Extrayendo audio de: {VIDEO_PATH}")
    clip = VideoFileClip(str(VIDEO_PATH))
    if clip.audio is None:
        print("[ERROR] El video no tiene pista de audio.")
        sys.exit(1)

    sr_audio = 22050
    audio_completo = clip.audio.to_soundarray(fps=sr_audio)
    duracion = clip.duration
    clip.close()

    print(f"[OK] Audio extraído. Duración: {duracion:.1f}s")
    print(f"[INFO] Iniciando envío al topic '{TOPIC_AUDIO}'...")
    print("-" * 50)

    ventana_s = 4.0
    salto_s = 2.0  # Procesamos y enviamos un fragmento de 4s cada 2s de video
    
    tiempo_actual = 0.0
    segmento_id = 0

    while tiempo_actual < duracion:
        inicio = max(0, tiempo_actual - ventana_s)
        fin = min(tiempo_actual, duracion)
        
        idx_inicio = int(inicio * sr_audio)
        idx_fin = int(fin * sr_audio)
        
        if idx_fin > idx_inicio:
            segmento = audio_completo[idx_inicio:idx_fin]
            clase, confianza = clasificar_audio(modelo_cnn, segmento, sr_audio)
            
            mensaje = {
                "segmento_id": segmento_id,
                "timestamp_video": tiempo_actual,
                "timestamp_sistema": time.time(),
                "clase_audio": clase,
                "confianza": confianza
            }
            
            productor.send(TOPIC_AUDIO, value=mensaje)
            print(f"  [Segmento {segmento_id:>3} | {tiempo_actual:>5.1f}s] {clase:<20} ({confianza:.2f})")
            segmento_id += 1

        tiempo_actual += salto_s
        time.sleep(salto_s / 3)  # Simulamos tiempo real acelerado (como el de video)

    productor.flush()
    print("-" * 50)
    print(f"[OK] Transmisión de audio finalizada. Total mensajes: {segmento_id}")

if __name__ == "__main__":
    procesar_audio()
