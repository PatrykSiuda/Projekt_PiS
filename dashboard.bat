@echo off
cd /d "%~dp0"
echo Uruchamianie dashboardu Streamlit...
echo Otworz przegladarke: http://localhost:8501
echo (aby zatrzymac: Ctrl+C)
echo.
py -m streamlit run dashboard.py
pause
