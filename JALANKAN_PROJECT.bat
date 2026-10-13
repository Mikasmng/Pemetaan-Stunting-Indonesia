@echo off
cd /d "%~dp0"

echo ==============================================
echo STUNTINGWATCH - SSGI 2024
echo ==============================================

echo.
echo [1/3] Mengecek dataset...
python 01_eksplorasi.py
if errorlevel 1 goto :error

echo.
echo [2/3] Melatih dan mengevaluasi model XGBoost...
python 02_model.py
if errorlevel 1 goto :error

echo.
echo [3/3] Mengekspor data dashboard...
python 04_export_data.py
if errorlevel 1 goto :error

echo.
echo Dashboard akan dibuka di http://localhost:8000
start "" http://localhost:8000
python -m http.server 8000 -d dashboard
goto :eof

:error
echo.
echo Terjadi error. Pastikan dependency sudah terpasang dengan:
echo python -m pip install -r requirements.txt
pause
