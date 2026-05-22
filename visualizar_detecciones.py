"""
Visualizador de Detecciones - Pipeline de Percepción Computacional
===================================================================
Genera un video con bounding boxes, etiquetas y conteo de vehículos
en tiempo real sobre el video de tráfico.

Adaptado 100% para Windows.
"""

import cv2
import time
import sys
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

# ─── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELO_YOLO = BASE_DIR / "modelos" / "yolov8_best.pt"
VIDEO_PATH = BASE_DIR / "data" / "trafico.mp4"
VIDEO_SALIDA = BASE_DIR / "data" / "trafico_detecciones.mp4"

# ─── Colores por clase (BGR) ────────────────────────────────────────────────
COLORES = {
    "car":        (0, 200, 255),    # Naranja
    "bus":        (0, 255, 0),      # Verde
    "truck":      (255, 100, 0),    # Azul
    "van":        (255, 255, 0),    # Cyan
    "pedestrian": (0, 0, 255),      # Rojo
    "people":     (0, 0, 200),      # Rojo oscuro
    "motor":      (255, 0, 255),    # Magenta
    "tricycle":   (200, 200, 0),    # Cyan oscuro
    "bicycle":    (255, 150, 50),   # Azul claro
}

COLOR_DEFAULT = (200, 200, 200)  # Gris para clases no definidas


def obtener_color(clase):
    return COLORES.get(clase, COLOR_DEFAULT)


def dibujar_panel_conteo(frame, conteo_frame, conteo_total, frame_id, total_frames, fps):
    """Dibuja un panel semitransparente con el conteo de objetos."""
    alto, ancho = frame.shape[:2]

    # ─── Panel superior: Info del frame ─────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (ancho, 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    texto_frame = f"Frame: {frame_id}/{total_frames}  |  FPS: {fps:.1f}  |  Objetos: {sum(conteo_frame.values())}"
    cv2.putText(frame, texto_frame, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ─── Panel lateral: Conteo por clase ────────────────────────
    panel_ancho = 280
    panel_alto = 40 + len(conteo_frame) * 35 + 60
    panel_y = 50

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (ancho - panel_ancho - 10, panel_y),
                  (ancho - 10, panel_y + panel_alto), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Título
    cv2.putText(frame, "CONTEO EN VIVO", (ancho - panel_ancho, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Conteo por clase
    y_offset = panel_y + 55
    for clase, cantidad in sorted(conteo_frame.items(), key=lambda x: x[1], reverse=True):
        color = obtener_color(clase)
        # Cuadro de color
        cv2.rectangle(frame, (ancho - panel_ancho, y_offset - 12),
                      (ancho - panel_ancho + 15, y_offset + 3), color, -1)
        # Texto
        texto = f"{clase}: {cantidad}"
        cv2.putText(frame, texto, (ancho - panel_ancho + 22, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 35

    # Total acumulado
    cv2.line(frame, (ancho - panel_ancho, y_offset - 5),
             (ancho - 20, y_offset - 5), (100, 100, 100), 1)
    total_obj = sum(conteo_total.values())
    cv2.putText(frame, f"TOTAL ACUMULADO: {total_obj}", (ancho - panel_ancho, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

    return frame


def procesar_video():
    """Procesa el video y genera la visualización con detecciones."""

    # Validar archivos
    if not MODELO_YOLO.exists():
        print(f"[ERROR] Modelo no encontrado: {MODELO_YOLO}")
        sys.exit(1)
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video no encontrado: {VIDEO_PATH}")
        sys.exit(1)

    # Cargar modelo
    print(f"[INFO] Cargando modelo YOLOv8 desde: {MODELO_YOLO}")
    modelo = YOLO(str(MODELO_YOLO))

    # Abrir video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {VIDEO_PATH}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video entrada: {ancho}x{alto} @ {fps_video:.1f} FPS, {total_frames} frames")

    # Crear video de salida
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_SALIDA), fourcc, fps_video, (ancho, alto))

    print(f"[INFO] Video salida: {VIDEO_SALIDA}")
    print(f"[INFO] Procesando... esto puede tomar unos minutos.")
    print("-" * 60)

    frame_id = 0
    conteo_total = defaultdict(int)
    tiempo_inicio = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Detección YOLOv8
            resultados = modelo(frame, verbose=False, conf=0.3)

            conteo_frame = defaultdict(int)

            for r in resultados:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confianza = float(box.conf[0])
                    clase_id = int(box.cls[0])
                    nombre_clase = modelo.names[clase_id]

                    conteo_frame[nombre_clase] += 1
                    conteo_total[nombre_clase] += 1

                    color = obtener_color(nombre_clase)

                    # Dibujar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Etiqueta con fondo
                    etiqueta = f"{nombre_clase} {confianza:.2f}"
                    (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, etiqueta, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Calcular FPS de procesamiento
            tiempo_transcurrido = time.time() - tiempo_inicio
            fps_proceso = frame_id / tiempo_transcurrido if tiempo_transcurrido > 0 else 0

            # Dibujar panel de conteo
            frame = dibujar_panel_conteo(frame, conteo_frame, conteo_total,
                                         frame_id, total_frames, fps_proceso)

            # Escribir frame al video de salida
            out.write(frame)

            # Progreso en consola
            if frame_id % 50 == 0:
                porcentaje = (frame_id / total_frames) * 100
                print(f"  Progreso: {frame_id}/{total_frames} ({porcentaje:.0f}%) | "
                      f"{sum(conteo_frame.values())} objetos en frame | "
                      f"FPS: {fps_proceso:.1f}")

    except KeyboardInterrupt:
        print("\n[INFO] Procesamiento interrumpido por el usuario.")
    finally:
        cap.release()
        out.release()

    # Reporte final
    tiempo_total = time.time() - tiempo_inicio
    print("-" * 60)
    print(f"[OK] Video generado: {VIDEO_SALIDA}")
    print(f"[OK] Frames procesados: {frame_id}")
    print(f"[OK] Tiempo de procesamiento: {tiempo_total:.1f} segundos")
    print(f"[OK] Velocidad promedio: {frame_id / tiempo_total:.1f} FPS")
    print()
    print("  Abre el video con:")
    print(f'  start "" "{VIDEO_SALIDA}"')


if __name__ == "__main__":
    print("=" * 60)
    print("  VISUALIZADOR DE DETECCIONES")
    print("  Pipeline de Percepción Computacional")
    print("=" * 60)
    procesar_video()
