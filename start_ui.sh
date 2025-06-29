#!/bin/bash

echo "Starting AIQ Toolkit UI..."
echo

# Check if external/aiqtoolkit-opensource-ui directory exists
if [ ! -d "external/aiqtoolkit-opensource-ui" ]; then
    echo "Error: external/aiqtoolkit-opensource-ui directory not found"
    echo "Make sure you are running this script from the project root directory"
    exit 1
fi

cd external/aiqtoolkit-opensource-ui

# Check if node_modules exists, if not run npm install
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
    echo
fi

echo "Starting UI on port 3000..."
echo
echo "The UI will be available at: http://localhost:3000"
echo
echo "Use the mode switcher in the UI to toggle between:"
echo "- ON_CALL mode (connects to backend on port 8000)"
echo "- FRIDAY mode (connects to backend on port 8001)"
echo "- SLACK mode (connects to backend on port 8002)"
echo

npm run dev 