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
KAFKA_BROKER = "localhost:9092"
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
        TOPIC_DETECCIONES,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="grupo-consumidor-spark",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=30000,  # 30 segundos sin mensajes = parar
    )

    print(f"[INFO] Consumiendo del topic '{TOPIC_DETECCIONES}'...")
    print(f"[INFO] Se detendrá automáticamente tras 30s sin mensajes nuevos.")
    print("-" * 70)

    # ─── Estadísticas acumuladas ────────────────────────────────────────
    total_frames = 0
    total_objetos = 0
    conteo_clases = defaultdict(int)
    confianza_acumulada = defaultdict(list)
    timestamp_inicio = None
    timestamp_ultimo = None

    try:
        for mensaje in consumer:
            data = mensaje.value

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
                f"  [Frame {frame_id:>5}] "
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

        # Guardar reporte en archivo
        reporte_path = BASE_DIR / "data" / "reporte_pipeline.txt"
        with open(reporte_path, "w", encoding="utf-8") as f:
            f.write("REPORTE FINAL DEL PIPELINE DE PERCEPCIÓN\n")
            f.write("=" * 50 + "\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Frames procesados: {total_frames}\n")
            f.write(f"Total de objetos detectados: {total_objetos}\n")
            f.write(f"Objetos promedio por frame: {total_objetos / total_frames:.1f}\n\n")
            f.write("DETECCIONES POR CLASE:\n")
            f.write("-" * 50 + "\n")
            for clase, conteo in sorted(
                conteo_clases.items(), key=lambda x: x[1], reverse=True
            ):
                porcentaje = (conteo / total_objetos * 100) if total_objetos > 0 else 0
                confs = confianza_acumulada[clase]
                avg_conf = sum(confs) / len(confs) if confs else 0
                f.write(f"  {clase}: {conteo} ({porcentaje:.1f}%) - confianza: {avg_conf:.4f}\n")

        print(f"\n  [OK] Reporte guardado en: {reporte_path}")
    else:
        print("  No se recibieron mensajes.")

    print("=" * 70)


# ─── Función opcional: Consumidor con PySpark ───────────────────────────────
def consumir_con_spark():
    """
    Versión alternativa usando PySpark Structured Streaming.
    Requiere configuración adicional de winutils en Windows.

    Para usar esta versión:
    1. Descargar winutils.exe de:
       https://github.com/steveloughran/winutils/tree/master/hadoop-3.0.0/bin
    2. Crear carpeta C:\\hadoop\\bin y colocar winutils.exe ahí
    3. Configurar variable de entorno HADOOP_HOME=C:\\hadoop
    4. Agregar C:\\hadoop\\bin al PATH
    5. Instalar pyspark: pip install pyspark
    """
    try:
        # Configurar Hadoop para Windows
        hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["PATH"] = os.environ["PATH"] + ";" + os.path.join(hadoop_home, "bin")

        from pyspark.sql import SparkSession
        from pyspark.sql.functions import from_json, col, explode, window
        from pyspark.sql.types import (
            StructType, StructField, StringType, FloatType,
            IntegerType, ArrayType, DoubleType, TimestampType,
        )

        # Esquema del JSON que envía el productor
        schema_bbox = StructType([
            StructField("x1", FloatType()),
            StructField("y1", FloatType()),
            StructField("x2", FloatType()),
            StructField("y2", FloatType()),
        ])

        schema_deteccion = StructType([
            StructField("clase", StringType()),
            StructField("clase_id", IntegerType()),
            StructField("confianza", FloatType()),
            StructField("bbox", schema_bbox),
        ])

        schema_mensaje = StructType([
            StructField("frame_id", IntegerType()),
            StructField("timestamp", DoubleType()),
            StructField("timestamp_legible", StringType()),
            StructField("total_objetos", IntegerType()),
            StructField("detecciones", ArrayType(schema_deteccion)),
        ])

        # Crear sesión Spark
        spark = (
            SparkSession.builder
            .appName("PipelinePercepcion-Consumidor")
            .master("local[*]")
            .config("spark.jars.packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")

        # Leer stream de Kafka
        df_kafka = (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BROKER)
            .option("subscribe", TOPIC_DETECCIONES)
            .option("startingOffsets", "earliest")
            .load()
        )

        # Parsear JSON
        df_parsed = (
            df_kafka
            .selectExpr("CAST(value AS STRING) as json_str")
            .select(from_json(col("json_str"), schema_mensaje).alias("data"))
            .select("data.*")
        )

        # Explotar detecciones y contar por clase
        df_detecciones = (
            df_parsed
            .select(
                col("frame_id"),
                col("timestamp"),
                explode(col("detecciones")).alias("det"),
            )
            .select(
                col("frame_id"),
                col("timestamp"),
                col("det.clase").alias("clase"),
                col("det.confianza").alias("confianza"),
            )
        )

        # Escribir a consola
        query = (
            df_detecciones
            .groupBy("clase")
            .count()
            .writeStream
            .outputMode("complete")
            .format("console")
            .option("truncate", "false")
            .trigger(processingTime="5 seconds")
            .start()
        )

        print("[INFO] Spark Streaming iniciado. Ctrl+C para detener.")
        query.awaitTermination()

    except ImportError:
        print("[ERROR] PySpark no está instalado. Usa: pip install pyspark")
        print("[INFO] Usando consumidor estándar con kafka-python...")
        consumir_y_analizar()
    except Exception as e:
        print(f"[ERROR] Error con Spark: {e}")
        print("[INFO] Usando consumidor estándar con kafka-python...")
        consumir_y_analizar()


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
