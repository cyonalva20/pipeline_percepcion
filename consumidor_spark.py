"""
Consumidor Spark Streaming - Pipeline de Percepción Computacional
=================================================================
Consume mensajes de Kafka con Spark Structured Streaming,
procesa las detecciones de tráfico y genera estadísticas en tiempo real.

Adaptado 100% para Windows (incluye configuración de winutils).
"""

import json
import time
import os
import sys
from pathlib import Path
from collections import defaultdict

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

# ─── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── Configuración Kafka ───────────────────────────────────────────────────
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
TOPIC_DETECCIONES = "detecciones-trafico"


def esperar_kafka(broker: str, max_intentos: int = 30, intervalo: int = 3):
    """Espera hasta que Kafka esté disponible."""
    for intento in range(1, max_intentos + 1):
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=broker,
                api_version_auto_timeout_ms=5000,
            )
            consumer.close()
            print(f"[OK] Kafka disponible en {broker}")
            return True
        except NoBrokersAvailable:
            print(f"[ESPERA] Kafka no disponible, intento {intento}/{max_intentos}...")
            time.sleep(intervalo)
    print("[ERROR] No se pudo conectar a Kafka.")
    sys.exit(1)


def consumir_y_analizar():
    """
    Consume mensajes del topic de detecciones y genera análisis en tiempo real.

    Nota: Usamos kafka-python directamente como consumidor inteligente.
    Esto evita la complejidad de configurar PySpark + winutils en Windows
    y cumple el mismo objetivo funcional del pipeline de streaming.
    Para un entorno de producción real se usaría Spark Structured Streaming.
    """

    esperar_kafka(KAFKA_BROKER)

    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="grupo-consumidor-spark",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=30000,  # 30 segundos sin mensajes = parar
    )
    consumer.subscribe([TOPIC_DETECCIONES, "audio-stream"])

    print(f"[INFO] Consumiendo de los topics '{TOPIC_DETECCIONES}' y 'audio-stream'...")
    print(f"[INFO] Se detendrá automáticamente tras 30s sin mensajes nuevos.")
    print("-" * 70)

    # ─── Estadísticas acumuladas ────────────────────────────────────────
    total_frames = 0
    total_objetos = 0
    conteo_clases = defaultdict(int)
    confianza_acumulada = defaultdict(list)
    timestamp_inicio = None
    timestamp_ultimo = None
    conteo_audio = defaultdict(int)
    confianza_audio = defaultdict(list)
    total_audio_mensajes = 0

    try:
        for mensaje in consumer:
            data = mensaje.value
            topic = mensaje.topic

            if topic == "audio-stream":
                total_audio_mensajes += 1
                clase_audio = data.get("clase_audio", "desconocido")
                conf_audio = data.get("confianza", 0)
                conteo_audio[clase_audio] += 1
                confianza_audio[clase_audio].append(conf_audio)
                
                print(f"  [AUDIO] Segmento {data.get('segmento_id', 0):>3} | {clase_audio:<15s} (conf: {conf_audio:.2f})")
                
                timestamp = data.get("timestamp_sistema", time.time())
                if timestamp_inicio is None: timestamp_inicio = timestamp
                timestamp_ultimo = timestamp

            elif topic == TOPIC_DETECCIONES:
                frame_id = data.get("frame_id", 0)
                detecciones = data.get("detecciones", [])
                timestamp = data.get("timestamp", time.time())
                n_objetos = data.get("total_objetos", len(detecciones))

                if timestamp_inicio is None:
                    timestamp_inicio = timestamp
                timestamp_ultimo = timestamp

                total_frames += 1
                total_objetos += n_objetos

                # Acumular estadísticas por clase
                for det in detecciones:
                    clase = det.get("clase", "desconocido")
                    conf = det.get("confianza", 0)
                    conteo_clases[clase] += 1
                    confianza_acumulada[clase].append(conf)

                # Mostrar resumen del frame
                clases_frame = defaultdict(int)
                for d in detecciones:
                    clases_frame[d.get("clase", "?")] += 1

                resumen = ", ".join(f"{v} {k}" for k, v in clases_frame.items())

                print(
                    f"  [VIDEO Frame {frame_id:>5}] "
                    f"{n_objetos:>3} objetos | {resumen}"
                )

            # Cada 20 frames, mostrar estadísticas acumuladas
            if total_frames % 20 == 0:
                print()
                print("  " + "=" * 50)
                print(f"  ESTADÍSTICAS ACUMULADAS ({total_frames} frames)")
                print("  " + "-" * 50)
                for clase, conteo in sorted(
                    conteo_clases.items(), key=lambda x: x[1], reverse=True
                ):
                    confs = confianza_acumulada[clase]
                    avg_conf = sum(confs) / len(confs) if confs else 0
                    print(
                        f"    {clase:<20s} | {conteo:>5} detecciones | "
                        f"confianza prom: {avg_conf:.3f}"
                    )
                print("  " + "=" * 50)
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Consumo interrumpido por el usuario.")
    finally:
        consumer.close()

    # ─── Reporte final ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  REPORTE FINAL DEL PIPELINE DE PERCEPCIÓN")
    print("=" * 70)

    if total_frames > 0:
        duracion = (timestamp_ultimo - timestamp_inicio) if timestamp_inicio and timestamp_ultimo else 0

        print(f"  Frames procesados:    {total_frames}")
        print(f"  Total de objetos:     {total_objetos}")
        print(f"  Objetos/frame (prom): {total_objetos / total_frames:.1f}")
        if duracion > 0:
            print(f"  Duración del video:   {duracion:.1f} segundos")
        print()
        print("  DETECCIONES POR CLASE:")
        print("  " + "-" * 55)
        print(f"  {'Clase':<20s} | {'Cantidad':>8} | {'% del total':>10} | {'Conf. prom':>10}")
        print("  " + "-" * 55)

        for clase, conteo in sorted(
            conteo_clases.items(), key=lambda x: x[1], reverse=True
        ):
            porcentaje = (conteo / total_objetos * 100) if total_objetos > 0 else 0
            confs = confianza_acumulada[clase]
            avg_conf = sum(confs) / len(confs) if confs else 0
            print(
                f"  {clase:<20s} | {conteo:>8} | {porcentaje:>9.1f}% | {avg_conf:>10.4f}"
            )
        print("  " + "-" * 55)

        if total_audio_mensajes > 0:
            print()
            print("  SONIDO URBANO (CNN):")
            print("  " + "-" * 55)
            print(f"  {'Clase Audio':<20s} | {'Segmentos':>9} | {'% del total':>10} | {'Conf. prom':>10}")
            print("  " + "-" * 55)
            for clase, conteo in sorted(conteo_audio.items(), key=lambda x: x[1], reverse=True):
                porcentaje = (conteo / total_audio_mensajes * 100)
                confs = confianza_audio[clase]
                avg_conf = sum(confs) / len(confs) if confs else 0
                print(f"  {clase:<20s} | {conteo:>9} | {porcentaje:>9.1f}% | {avg_conf:>10.4f}")
            print("  " + "-" * 55)

        # Guardar reporte en archivo
        reporte_path = BASE_DIR / "data" / "reporte_pipeline.txt"
        with open(reporte_path, "w", encoding="utf-8") as f:
            f.write("REPORTE FINAL DEL PIPELINE MULTIMODAL (VIDEO + AUDIO)\n")
            f.write("=" * 55 + "\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Frames procesados: {total_frames}\n")
            f.write(f"Total de objetos detectados: {total_objetos}\n")
            f.write(f"Objetos promedio por frame: {total_objetos / total_frames:.1f}\n")
            if duracion > 0: f.write(f"Duración: {duracion:.1f} segundos\n")
            f.write("\n1. DETECCIONES VEHICULARES (YOLOv8):\n")
            f.write("-" * 55 + "\n")
            for clase, conteo in sorted(
                conteo_clases.items(), key=lambda x: x[1], reverse=True
            ):
                porcentaje = (conteo / total_objetos * 100) if total_objetos > 0 else 0
                confs = confianza_acumulada[clase]
                avg_conf = sum(confs) / len(confs) if confs else 0
                f.write(f"  {clase}: {conteo} ({porcentaje:.1f}%) - confianza: {avg_conf:.4f}\n")
                
            if total_audio_mensajes > 0:
                f.write("\n2. CLASIFICACIÓN DE AUDIO (CNN):\n")
                f.write("-" * 55 + "\n")
                for clase, conteo in sorted(conteo_audio.items(), key=lambda x: x[1], reverse=True):
                    porcentaje = (conteo / total_audio_mensajes * 100)
                    confs = confianza_audio[clase]
                    avg_conf = sum(confs) / len(confs) if confs else 0
                    f.write(f"  {clase}: {conteo} segmentos ({porcentaje:.1f}%) - confianza: {avg_conf:.4f}\n")

        print(f"\n  [OK] Reporte guardado en: {reporte_path}")
    else:
        print("  No se recibieron mensajes.")

    print("=" * 70)


# ─── Función opcional: Consumidor con PySpark ───────────────────────────────
def consumir_con_spark():
    """
    Versión oficial y corregida usando PySpark Structured Streaming.
    Procesa de forma nativa y paralela los streams de Video y Audio.
    """
    try:
        # 1. Configuración del entorno (Se mantiene por compatibilidad local)
        hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["PATH"] = os.environ["PATH"] + ";" + os.path.join(hadoop_home, "bin")

        from pyspark.sql import SparkSession  # type: ignore
        from pyspark.sql.functions import from_json, col, explode  # type: ignore
        from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType, DoubleType  # type: ignore
        import threading

        # 2. Inicializar la sesión de Spark
        spark = (
            SparkSession.builder
            .appName("PipelinePercepcion-Consumidor-Multimodal")
            .master("local[*]")
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
            .getOrCreate()
        )
        spark.snarkContext.setLogLevel("ERROR") if hasattr(spark, 'snarkContext') else spark.sparkContext.setLogLevel("ERROR")

        # ─── DEFINICIÓN DE ESQUEMAS ─────────────────────────────────────────
        # Esquema para las detecciones de Video (YOLOv8)
        schema_bbox = StructType([
            StructField("x1", FloatType()), StructField("y1", FloatType()),
            StructField("x2", FloatType()), StructField("y2", FloatType())
        ])
        schema_deteccion = StructType([
            StructField("clase", StringType()),
            StructField("clase_id", IntegerType()),
            StructField("confianza", FloatType()),
            StructField("bbox", schema_bbox)
        ])
        schema_video = StructType([
            StructField("frame_id", IntegerType()),
            StructField("timestamp", DoubleType()),
            StructField("timestamp_legible", StringType()),
            StructField("total_objetos", IntegerType()),
            StructField("detecciones", ArrayType(schema_deteccion))
        ])

        # Esquema para las detecciones de Audio (CNN) - Coincide exactamente con tu productor_audio.py
        schema_audio = StructType([
            StructField("segmento_id", IntegerType()),
            StructField("timestamp_video", DoubleType()),
            StructField("timestamp_sistema", DoubleType()),
            StructField("clase_audio", StringType()),
            StructField("confianza", FloatType())
        ])

        # 3. Lectura unificada de Kafka suscribiéndose a AMBOS tópicos
        df_kafka_raiz = (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BROKER)
            .option("subscribe", f"{TOPIC_DETECCIONES},audio-stream") # Suscripción simultánea
            .option("startingOffsets", "latest")
            .load()
        )

        # Convertir datos crudos a String
        df_strings = df_kafka_raiz.selectExpr("CAST(topic AS STRING) as item_topic", "CAST(value AS STRING) as json_str")

        # 4. Separación y Deserialización de Flujos mediante filtrado de Tópicos
        # Flujo A: Video
        df_video_parsed = (
            df_strings.filter(col("item_topic") == TOPIC_DETECCIONES)
            .select(from_json(col("json_str"), schema_video).alias("data"))
            .select("data.*")
        )
        
        # Flujo B: Audio
        df_audio_parsed = (
            df_strings.filter(col("item_topic") == "audio-stream")
            .select(from_json(col("json_str"), schema_audio).alias("data"))
            .select("data.*")
        )

        # 5. Preparación de Métricas Continuas
        df_conteo_vehicular = (
            df_video_parsed
            .select(explode(col("detecciones")).alias("det"))
            .groupBy("det.clase")
            .count()
        )

        df_conteo_acustico = (
            df_audio_parsed
            .groupBy("clase_audio")
            .count()
        )

        print("\n" + "="*70)
        print(" 🚀 ENTORNO PYSPARK MULTIMODAL INICIADO CORRECTAMENTE")
        print(" Escuchando transmisiones en vivo de Video y Audio... (Ctrl+C para salir)")
        print("="*70 + "\n")

        # --- GENERACIÓN DEL REPORTE UNIFICADO ---
        estado_global = {"video": {}, "audio": {}}
        ruta_reporte = BASE_DIR / "data" / "reporte_pipeline.txt"
        lock_reporte = threading.Lock()

        def guardar_reporte_txt():
            """Sobreescribe el reporte unificado en disco en cada micro-batch."""
            with lock_reporte:
                ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
                total_vehiculos = sum(estado_global["video"].values())
                total_audios = sum(estado_global["audio"].values())
                
                with open(ruta_reporte, "w", encoding="utf-8") as f:
                    f.write("REPORTE FINAL DEL PIPELINE MULTIMODAL (VIDEO + AUDIO) - PYSPARK\n")
                    f.write("=" * 65 + "\n")
                    f.write(f"Última actualización: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("1. DETECCIONES VEHICULARES (YOLOv8):\n")
                    f.write("-" * 65 + "\n")
                    if not estado_global["video"]:
                        f.write("  (Esperando datos de video...)\n")
                    for clase, conteo in sorted(estado_global["video"].items(), key=lambda x: x[1], reverse=True):
                        pct = (conteo / total_vehiculos * 100) if total_vehiculos > 0 else 0
                        f.write(f"  {clase:<15s}: {conteo:>5} detecciones ({pct:>5.1f}%)\n")
                        
                    f.write("\n2. CLASIFICACIÓN DE AUDIO (CNN):\n")
                    f.write("-" * 65 + "\n")
                    if not estado_global["audio"]:
                        f.write("  (Esperando datos de audio...)\n")
                    for clase, conteo in sorted(estado_global["audio"].items(), key=lambda x: x[1], reverse=True):
                        pct = (conteo / total_audios * 100) if total_audios > 0 else 0
                        f.write(f"  {clase:<15s}: {conteo:>5} segmentos   ({pct:>5.1f}%)\n")

        def procesar_batch_video(df_batch, batch_id):
            print(f"\n📊 [MÉTRICA VIDEO] CONTEO DE VEHÍCULOS (YOLOv8) - Batch {batch_id}")
            df_batch.show(truncate=False)
            filas = df_batch.collect()
            estado_global["video"] = {fila["clase"]: fila["count"] for fila in filas}
            guardar_reporte_txt()

        def procesar_batch_audio(df_batch, batch_id):
            print(f"\n🔊 [MÉTRICA AUDIO] CLASIFICACIÓN ACÚSTICA URBANA (CNN) - Batch {batch_id}")
            df_batch.show(truncate=False)
            filas = df_batch.collect()
            estado_global["audio"] = {fila["clase_audio"]: fila["count"] for fila in filas}
            guardar_reporte_txt()

        # 6. Lanzamiento de las dos consultas en paralelo (Multi-Query Sink con Custom ForeachBatch)
        query_video = (
            df_conteo_vehicular.writeStream
            .outputMode("complete")
            .foreachBatch(procesar_batch_video)
            .option("checkpointLocation", str(BASE_DIR / "checkpoints" / "video"))
            .trigger(processingTime="3 seconds")
            .start()
        )

        query_audio = (
            df_conteo_acustico.writeStream
            .outputMode("complete")
            .foreachBatch(procesar_batch_audio)
            .option("checkpointLocation", str(BASE_DIR / "checkpoints" / "audio"))
            .trigger(processingTime="3 seconds")
            .start()
        )

        # Mantener vivos ambos streams simultáneamente
        query_video.awaitTermination()
        query_audio.awaitTermination()

    except Exception as e:
        print(f"\n[ERROR CRÍTICO EN ENTORNO SPARK]: {e}")
        print(" Abortando ejecución segura. Por favor, verifique dependencias.")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 70)
    print("  CONSUMIDOR - Pipeline de Percepción")
    print("  Punto 6: Arquitectura de Datos y Pipeline a Escala")
    print("=" * 70)
    print()

    # Verificar argumento para elegir modo
    if len(sys.argv) > 1 and sys.argv[1] == "--spark":
        print("[MODO] Spark Structured Streaming")
        consumir_con_spark()
    else:
        print("[MODO] Consumidor Kafka estándar (recomendado para Windows)")
        print("[TIP]  Usa --spark para el modo PySpark (requiere winutils)")
        print()
        consumir_y_analizar()
