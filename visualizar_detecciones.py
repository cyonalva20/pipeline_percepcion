"""
Visualizador de Detecciones - Pipeline de Percepción Computacional
===================================================================
Genera un video con:
  - YOLOv8: Bounding boxes y conteo de vehículos en tiempo real
  - CNN Audio: Clasificación de sonido urbano en tiempo real

Adaptado 100% para Windows.
"""

import cv2
import time
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import librosa
from moviepy import VideoFileClip
from ultralytics import YOLO

# ─── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELO_YOLO = BASE_DIR / "modelos" / "yolov8_best.pt"
MODELO_CNN = BASE_DIR / "modelos" / "cnn_audio_urbansound.pt"
VIDEO_PATH = BASE_DIR / "data" / "trafico.mp4"
VIDEO_SALIDA = BASE_DIR / "data" / "trafico_detecciones.mp4"

# ─── Clases de UrbanSound8K ────────────────────────────────────────────────
CLASES_AUDIO = [
    "Aire acondicionado",   # 0 - air_conditioner
    "Bocina de auto",       # 1 - car_horn
    "Niños jugando",        # 2 - children_playing
    "Ladrido de perro",     # 3 - dog_bark
    "Taladro",              # 4 - drilling
    "Motor en marcha",      # 5 - engine_idling
    "Disparo",              # 6 - gun_shot
    "Martillo neumático",   # 7 - jackhammer
    "Sirena",               # 8 - siren
    "Música callejera",     # 9 - street_music
]

# ─── Colores por clase vehicular (BGR) ──────────────────────────────────────
COLORES = {
    "car":        (0, 200, 255),
    "bus":        (0, 255, 0),
    "truck":      (255, 100, 0),
    "van":        (255, 255, 0),
    "pedestrian": (0, 0, 255),
    "people":     (0, 0, 200),
    "motor":      (255, 0, 255),
    "tricycle":   (200, 200, 0),
    "bicycle":    (255, 150, 50),
}

COLOR_DEFAULT = (200, 200, 200)


# ─── Definición de la CNN de Audio (misma arquitectura del entrenamiento) ───
class CNNAudioUrbano(nn.Module):
    """
    CNN para clasificación de sonido urbano.
    Arquitectura: 3 bloques Conv2d+BN+ReLU+MaxPool → Flatten → FC(256) → FC(10)
    Entrada: espectrograma mel de 1x128x173 (n_mels=128, ~4s audio, hop=512)
    """
    def __init__(self, num_clases=10):
        super().__init__()
        self.bloque1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bloque2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bloque3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 21, 256),  # 43008 → 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_clases),
        )

    def forward(self, x):
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.clasificador(x)
        return x


def obtener_color(clase):
    return COLORES.get(clase, COLOR_DEFAULT)


def cargar_modelo_cnn(ruta):
    """Carga el modelo CNN de audio."""
    modelo = CNNAudioUrbano(num_clases=10)
    state_dict = torch.load(str(ruta), map_location="cpu", weights_only=False)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo


def clasificar_audio(modelo_cnn, audio_array, sr=22050):
    """
    Clasifica un fragmento de audio usando la CNN.
    Retorna (clase, confianza).
    """
    try:
        # Convertir a mono si es estéreo
        if len(audio_array.shape) > 1:
            audio_mono = np.mean(audio_array, axis=1)
        else:
            audio_mono = audio_array

        audio_mono = audio_mono.astype(np.float32)

        # Asegurar que el audio tenga exactamente 4 segundos (88200 muestras)
        target_len = sr * 4
        if len(audio_mono) < target_len:
            audio_mono = np.pad(audio_mono, (0, target_len - len(audio_mono)))
        else:
            audio_mono = audio_mono[:target_len]

        # Generar espectrograma mel (debe dar shape 128x173)
        mel = librosa.feature.melspectrogram(
            y=audio_mono, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Asegurar dimensión temporal exacta de 173
        if mel_db.shape[1] > 173:
            mel_db = mel_db[:, :173]
        elif mel_db.shape[1] < 173:
            mel_db = np.pad(mel_db, ((0, 0), (0, 173 - mel_db.shape[1])))

        # Normalizar
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        # Convertir a tensor: [1, 1, 128, 173]
        tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)

        # Inferencia
        with torch.no_grad():
            salida = modelo_cnn(tensor)
            probabilidades = torch.softmax(salida, dim=1)
            confianza, clase_idx = torch.max(probabilidades, dim=1)

        return CLASES_AUDIO[clase_idx.item()], confianza.item()

    except Exception as e:
        return "Sin clasificar", 0.0


def dibujar_panel_conteo(frame, conteo_frame, conteo_total, frame_id,
                          total_frames, fps, clase_audio, conf_audio):
    """Dibuja paneles de conteo vehicular y clasificación de audio."""
    alto, ancho = frame.shape[:2]

    # ─── Panel superior: Info del frame ─────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (ancho, 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    texto_frame = (
        f"Frame: {frame_id}/{total_frames}  |  "
        f"FPS: {fps:.1f}  |  "
        f"Objetos: {sum(conteo_frame.values())}"
    )
    cv2.putText(frame, texto_frame, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ─── Panel lateral derecho: Conteo vehicular ────────────────
    panel_ancho = 280
    n_clases = max(len(conteo_frame), 1)
    panel_alto = 40 + n_clases * 35 + 60
    panel_y = 50

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (ancho - panel_ancho - 10, panel_y),
                  (ancho - 10, panel_y + panel_alto), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "CONTEO EN VIVO", (ancho - panel_ancho, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_offset = panel_y + 55
    for clase, cantidad in sorted(conteo_frame.items(), key=lambda x: x[1], reverse=True):
        color = obtener_color(clase)
        cv2.rectangle(frame, (ancho - panel_ancho, y_offset - 12),
                      (ancho - panel_ancho + 15, y_offset + 3), color, -1)
        cv2.putText(frame, f"{clase}: {cantidad}",
                    (ancho - panel_ancho + 22, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 35

    cv2.line(frame, (ancho - panel_ancho, y_offset - 5),
             (ancho - 20, y_offset - 5), (100, 100, 100), 1)
    total_obj = sum(conteo_total.values())
    cv2.putText(frame, f"TOTAL ACUMULADO: {total_obj}",
                (ancho - panel_ancho, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

    # ─── Panel inferior izquierdo: Clasificación de audio ───────
    audio_panel_w = 350
    audio_panel_h = 70
    audio_panel_y = alto - audio_panel_h - 15
    audio_panel_x = 10

    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (audio_panel_x, audio_panel_y),
                  (audio_panel_x + audio_panel_w, audio_panel_y + audio_panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay3, 0.75, frame, 0.25, 0, frame)

    # Icono y título
    cv2.putText(frame, "SONIDO URBANO (CNN)",
                (audio_panel_x + 10, audio_panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # Clase detectada
    cv2.putText(frame, f"{clase_audio}",
                (audio_panel_x + 10, audio_panel_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Barra de confianza
    barra_x = audio_panel_x + 230
    barra_y = audio_panel_y + 35
    barra_ancho = 100
    barra_alto = 18
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + barra_ancho, barra_y + barra_alto),
                  (80, 80, 80), -1)
    fill_w = int(barra_ancho * conf_audio)
    # Color de la barra según confianza
    if conf_audio > 0.7:
        color_barra = (0, 255, 0)
    elif conf_audio > 0.4:
        color_barra = (0, 255, 255)
    else:
        color_barra = (0, 100, 255)
    cv2.rectangle(frame, (barra_x, barra_y),
                  (barra_x + fill_w, barra_y + barra_alto),
                  color_barra, -1)
    cv2.putText(frame, f"{conf_audio:.0%}",
                (barra_x + 35, barra_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def procesar_video():
    """Procesa el video con YOLOv8 (visual) + CNN (audio) y genera visualización."""

    # Validar archivos
    if not MODELO_YOLO.exists():
        print(f"[ERROR] Modelo YOLO no encontrado: {MODELO_YOLO}")
        sys.exit(1)
    if not MODELO_CNN.exists():
        print(f"[ERROR] Modelo CNN no encontrado: {MODELO_CNN}")
        sys.exit(1)
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video no encontrado: {VIDEO_PATH}")
        sys.exit(1)

    # ─── Cargar modelos ─────────────────────────────────────────
    print(f"[INFO] Cargando modelo YOLOv8 desde: {MODELO_YOLO}")
    modelo_yolo = YOLO(str(MODELO_YOLO))

    print(f"[INFO] Cargando modelo CNN Audio desde: {MODELO_CNN}")
    modelo_cnn = cargar_modelo_cnn(MODELO_CNN)
    print(f"[OK] Modelo CNN cargado ({sum(p.numel() for p in modelo_cnn.parameters())} parámetros)")

    # ─── Extraer audio del video ────────────────────────────────
    print(f"[INFO] Extrayendo audio del video...")
    clip = VideoFileClip(str(VIDEO_PATH))

    tiene_audio = clip.audio is not None
    audio_completo = None
    sr_audio = 22050

    if tiene_audio:
        audio_completo = clip.audio.to_soundarray(fps=sr_audio)
        print(f"[OK] Audio extraído: {audio_completo.shape[0]} muestras, "
              f"{audio_completo.shape[0]/sr_audio:.1f}s, "
              f"{audio_completo.shape[1]} canales")
    else:
        print("[AVISO] El video no tiene audio. Se mostrará 'Sin audio'.")

    duracion_video = clip.duration
    clip.close()

    # ─── Abrir video con OpenCV ─────────────────────────────────
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {VIDEO_PATH}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {ancho}x{alto} @ {fps_video:.1f} FPS, {total_frames} frames")

    # Video de salida
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_SALIDA), fourcc, fps_video, (ancho, alto))

    print(f"[INFO] Video salida: {VIDEO_SALIDA}")
    print(f"[INFO] Procesando con YOLOv8 + CNN Audio...")
    print("-" * 60)

    frame_id = 0
    conteo_total = defaultdict(int)
    tiempo_inicio = time.time()

    # Variables para audio (se actualiza cada ~2 segundos)
    clase_audio_actual = "Analizando..."
    conf_audio_actual = 0.0
    audio_update_interval = int(fps_video * 2)  # Cada 2 segundos de video
    ventana_audio = 4.0  # 4 segundos de audio para la CNN

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # ─── YOLO: Detección visual ─────────────────────────
            resultados = modelo_yolo(frame, verbose=False, conf=0.3)
            conteo_frame = defaultdict(int)

            for r in resultados:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confianza = float(box.conf[0])
                    clase_id = int(box.cls[0])
                    nombre_clase = modelo_yolo.names[clase_id]

                    conteo_frame[nombre_clase] += 1
                    conteo_total[nombre_clase] += 1

                    color = obtener_color(nombre_clase)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    etiqueta = f"{nombre_clase} {confianza:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                    )
                    cv2.rectangle(frame, (x1, y1 - th - 8),
                                  (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, etiqueta, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # ─── CNN: Clasificación de audio cada 2 segundos ────
            if tiene_audio and frame_id % audio_update_interval == 1:
                # Calcular posición temporal del frame actual
                tiempo_actual = frame_id / fps_video
                inicio_audio = max(0, tiempo_actual - ventana_audio)
                fin_audio = min(tiempo_actual, duracion_video)

                # Extraer segmento de audio
                idx_inicio = int(inicio_audio * sr_audio)
                idx_fin = int(fin_audio * sr_audio)

                if idx_fin > idx_inicio and idx_fin <= len(audio_completo):
                    segmento = audio_completo[idx_inicio:idx_fin]
                    clase_audio_actual, conf_audio_actual = clasificar_audio(
                        modelo_cnn, segmento, sr_audio
                    )

            # ─── Calcular FPS ───────────────────────────────────
            tiempo_transcurrido = time.time() - tiempo_inicio
            fps_proceso = frame_id / tiempo_transcurrido if tiempo_transcurrido > 0 else 0

            # ─── Dibujar paneles ────────────────────────────────
            frame = dibujar_panel_conteo(
                frame, conteo_frame, conteo_total,
                frame_id, total_frames, fps_proceso,
                clase_audio_actual, conf_audio_actual
            )

            out.write(frame)

            # Progreso
            if frame_id % 50 == 0:
                porcentaje = (frame_id / total_frames) * 100
                print(
                    f"  Progreso: {frame_id}/{total_frames} ({porcentaje:.0f}%) | "
                    f"{sum(conteo_frame.values())} objetos | "
                    f"Audio: {clase_audio_actual} ({conf_audio_actual:.0%}) | "
                    f"FPS: {fps_proceso:.1f}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")
    finally:
        cap.release()
        out.release()

    # Reporte final
    tiempo_total = time.time() - tiempo_inicio
    print("-" * 60)
    print(f"[OK] Video procesado: {frame_id} frames en {tiempo_total:.1f}s ({frame_id / tiempo_total:.1f} FPS)")

    # ─── Agregar audio original al video generado ───────────────
    if tiene_audio:
        print("[INFO] Agregando audio original al video...")
        try:
            video_sin_audio = VideoFileClip(str(VIDEO_SALIDA))
            audio_original = VideoFileClip(str(VIDEO_PATH)).audio

            video_con_audio = video_sin_audio.with_audio(audio_original)

            video_final_path = BASE_DIR / "data" / "trafico_detecciones_final.mp4"
            video_con_audio.write_videofile(
                str(video_final_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )

            video_sin_audio.close()
            video_con_audio.close()

            # Eliminar el video mudo temporal
            try:
                os.remove(str(VIDEO_SALIDA))
                # Renombrar el video final para que quede con el nombre original
                os.rename(str(video_final_path), str(VIDEO_SALIDA))
            except Exception as e:
                pass # Si falla el renombrado, dejamos el archivo _final

            print(f"[OK] Video CON audio: {VIDEO_SALIDA}")
            print()
            print("  Abre el video con:")
            print(f'  start "" "{VIDEO_SALIDA}"')
        except Exception as e:
            print(f"[AVISO] No se pudo agregar audio: {e}")
            print(f"[OK] Video sin audio disponible en: {VIDEO_SALIDA}")
    else:
        print(f"[OK] Video (sin audio): {VIDEO_SALIDA}")
        print()
        print("  Abre el video con:")
        print(f'  start "" "{VIDEO_SALIDA}"')


if __name__ == "__main__":
    print("=" * 60)
    print("  VISUALIZADOR DE DETECCIONES")
    print("  YOLOv8 (Video) + CNN (Audio)")
    print("  Pipeline de Percepción Computacional")
    print("=" * 60)
    procesar_video()
