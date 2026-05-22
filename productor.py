"""
Productor Kafka - Pipeline de Percepción Computacional
======================================================
Lee el video de tráfico (data/trafico.mp4), ejecuta detección con YOLOv8
y envía los resultados frame a frame al topic de Kafka.

Adaptado 100% para Windows.
"""

import json
import time
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ─── Rutas (relativas al script) ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELO_YOLO = BASE_DIR / "modelos" / "yolov8_best.pt"
VIDEO_PATH = BASE_DIR / "data" / "trafico.mp4"

# ─── Configuración Kafka ───────────────────────────────────────────────────
KAFKA_BROKER = "localhost:9092"
TOPIC_DETECCIONES = "detecciones-trafico"
TOPIC_FRAMES = "frames-info"


def esperar_kafka(broker: str, max_intentos: int = 30, intervalo: int = 3):
    """Espera hasta que Kafka esté disponible."""
    for intento in range(1, max_intentos + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=broker,
                api_version_auto_timeout_ms=5000,
            )
            producer.close()
            print(f"[OK] Kafka disponible en {broker}")
            return True
        except NoBrokersAvailable:
            print(
                f"[ESPERA] Kafka no disponible, intento {intento}/{max_intentos}..."
            )
            time.sleep(intervalo)
    print("[ERROR] No se pudo conectar a Kafka después de varios intentos.")
    sys.exit(1)


def crear_productor(broker: str) -> KafkaProducer:
    """Crea y retorna un productor Kafka con serialización JSON."""
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def procesar_video():
    """Procesa el video frame a frame con YOLOv8 y envía resultados a Kafka."""

    # Validar archivos
    if not MODELO_YOLO.exists():
        print(f"[ERROR] Modelo no encontrado: {MODELO_YOLO}")
        sys.exit(1)
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video no encontrado: {VIDEO_PATH}")
        sys.exit(1)

    # Cargar modelo YOLOv8
    print(f"[INFO] Cargando modelo YOLOv8 desde: {MODELO_YOLO}")
    modelo = YOLO(str(MODELO_YOLO))

    # Esperar a Kafka y crear productor
    esperar_kafka(KAFKA_BROKER)
    producer = crear_productor(KAFKA_BROKER)

    # Abrir video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {VIDEO_PATH}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {ancho}x{alto} @ {fps_video:.1f} FPS, {total_frames} frames")
    print(f"[INFO] Enviando detecciones al topic '{TOPIC_DETECCIONES}'...")
    print("-" * 60)

    frame_id = 0
    # Procesar 1 de cada 5 frames para no saturar
    SKIP_FRAMES = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Saltar frames para eficiencia
            if frame_id % SKIP_FRAMES != 0:
                continue

            timestamp = time.time()

            # Ejecutar detección YOLOv8
            resultados = modelo(frame, verbose=False, conf=0.3)

            detecciones = []
            for r in resultados:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confianza = float(box.conf[0])
                    clase_id = int(box.cls[0])
                    nombre_clase = modelo.names[clase_id]

                    detecciones.append({
                        "clase": nombre_clase,
                        "clase_id": clase_id,
                        "confianza": round(confianza, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                    })

            # Construir mensaje
            mensaje = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "timestamp_legible": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
                ),
                "resolucion": {"ancho": ancho, "alto": alto},
                "total_objetos": len(detecciones),
                "detecciones": detecciones,
            }

            # Enviar a Kafka
            producer.send(
                TOPIC_DETECCIONES,
                key=f"frame-{frame_id}",
                value=mensaje,
            )

            # Resumen en consola
            clases_detectadas = {}
            for d in detecciones:
                clases_detectadas[d["clase"]] = (
                    clases_detectadas.get(d["clase"], 0) + 1
                )

            resumen = ", ".join(
                f"{v} {k}" for k, v in clases_detectadas.items()
            )
            print(
                f"  Frame {frame_id:>5}/{total_frames} | "
                f"{len(detecciones):>3} objetos | {resumen}"
            )

            # Enviar info del frame al segundo topic
            producer.send(
                TOPIC_FRAMES,
                key=f"info-{frame_id}",
                value={
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "procesado": True,
                    "objetos_detectados": len(detecciones),
                },
            )

            producer.flush()

    except KeyboardInterrupt:
        print("\n[INFO] Producción interrumpida por el usuario.")
    finally:
        cap.release()
        producer.flush()
        producer.close()
        print("-" * 60)
        print(f"[FIN] Se procesaron {frame_id} frames del video.")


if __name__ == "__main__":
    print("=" * 60)
    print("  PRODUCTOR KAFKA - Pipeline de Percepción")
    print("  Punto 6: Arquitectura de Datos y Pipeline a Escala")
    print("=" * 60)
    procesar_video()
