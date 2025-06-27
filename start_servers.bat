@echo off
echo Starting AIQ Toolkit Backend Servers...
echo.

echo Starting ON_CALL backend server on port 8000...
start "ON_CALL Backend" cmd /k "aiq serve --config_file=examples/pdf_rag_chatbot/configs/incident_rag_chatbot.yml --port=8000"

echo Starting FRIDAY backend server on port 8001...
start "FRIDAY Backend" cmd /k "aiq serve --config_file=examples/friday/configs/friday_config.yml --port=8001"

echo Starting SLACK backend server on port 8002...
start "SLACK Backend" cmd /k "aiq serve --config_file=examples/pdf_rag_ingest/config/pdf_ingest_config.yml --port=8002"

echo.
echo Both backend servers are starting...
echo.
echo Backend servers:
echo ON_CALL backend: http://localhost:8000
echo FRIDAY backend: http://localhost:8001
echo SLACK backend: http://localhost:8002
echo.
echo Frontend UI:
echo Single UI instance: http://localhost:3000
echo (Use mode switcher to toggle between ON_CALL, FRIDAY, and SLACK backends)
echo.
echo To start the UI, run: npm run dev (in external/aiqtoolkit-opensource-ui)
echo.
echo Press any key to exit...
pause >nul 