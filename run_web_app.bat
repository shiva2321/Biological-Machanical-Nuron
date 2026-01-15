@echo off
REM Nuron Web App Launcher
REM Launches the Streamlit web interface correctly

echo.
echo ========================================
echo   Nuron Framework - Web Interface
echo ========================================
echo.
echo Starting Streamlit server...
echo.
echo The app will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run web_app.py

pause

