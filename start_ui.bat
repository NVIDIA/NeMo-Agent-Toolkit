@echo off
echo Starting AIQ Toolkit UI...
echo.

cd external\aiqtoolkit-opensource-ui

echo Starting UI on port 3000...
echo.
echo The UI will be available at: http://localhost:3000
echo.
echo Use the mode switcher in the UI to toggle between:
echo - ON_CALL mode (connects to backend on port 8000)
echo - FRIDAY mode (connects to backend on port 8001)
echo.

npm run dev 