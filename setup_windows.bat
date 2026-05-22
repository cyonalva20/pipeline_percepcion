@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo   SETUP - Pipeline de Percepcion Computacional (Windows)
echo ============================================================
echo.

REM ─── Verificar Python ─────────────────────────────────────────
echo [1/4] Verificando Python...
py --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.13 desde python.org
    echo         Asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)
echo [OK] Python encontrado:
py --version
echo.

REM ─── Crear entorno virtual ────────────────────────────────────
echo [2/4] Creando entorno virtual...
if exist venv (
    echo [INFO] Entorno virtual ya existe. Eliminando y recreando...
    rmdir /s /q venv
)
py -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo [OK] Entorno virtual creado en venv\
echo.

REM ─── Activar e instalar dependencias ──────────────────────────
echo [3/4] Instalando dependencias...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install kafka-python-ng ultralytics opencv-python numpy pyspark torch torchvision torchaudio

echo.
echo [OK] Dependencias instaladas.
echo.

REM ─── Verificar Docker ─────────────────────────────────────────
echo [4/4] Verificando Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Docker no encontrado en PATH.
    echo         Asegurate de que Docker Desktop este instalado y corriendo.
) else (
    echo [OK] Docker encontrado:
    docker --version
)

echo.
echo ============================================================
echo   SETUP COMPLETADO
echo ============================================================
echo.
echo   Para usar el pipeline:
echo   1. Abre CMD en esta carpeta
echo   2. Activa el entorno: venv\Scripts\activate.bat
echo   3. Inicia Kafka: docker compose up -d
echo   4. Ejecuta productor: python productor.py
echo   5. En otra terminal: python consumidor_spark.py
echo.
pause
