#!/bin/bash

echo "Starting AIQ Toolkit Backend Servers..."
echo

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Warning: Port $1 is already in use"
        return 1
    fi
    return 0
}

# Check if ports are available
echo "Checking if ports are available..."
check_port 8000
check_port 8001
check_port 8002
echo

echo "Starting ON_CALL backend server on port 8000..."
nohup aiq serve --config_file=examples/pdf_rag_chatbot/configs/incident_rag_chatbot.yml --port=8000 > logs/oncall_backend.log 2>&1 &
ONCALL_PID=$!
echo "ON_CALL backend started with PID: $ONCALL_PID"

echo "Starting FRIDAY backend server on port 8001..."
nohup aiq serve --config_file=examples/friday/configs/friday_config.yml --port=8001 > logs/friday_backend.log 2>&1 &
FRIDAY_PID=$!
echo "FRIDAY backend started with PID: $FRIDAY_PID"

echo "Starting SLACK backend server on port 8002..."
nohup aiq serve --config_file=examples/pdf_rag_ingest/config/pdf_ingest_config.yml --port=8002 > logs/slack_backend.log 2>&1 &
SLACK_PID=$!
echo "SLACK backend started with PID: $SLACK_PID"

# Create a PID file for easy cleanup
mkdir -p logs
echo "$ONCALL_PID" > logs/oncall_backend.pid
echo "$FRIDAY_PID" > logs/friday_backend.pid
echo "$SLACK_PID" > logs/slack_backend.pid

echo
echo "All backend servers are starting..."
echo
echo "Backend servers:"
echo "ON_CALL backend: http://localhost:8000"
echo "FRIDAY backend: http://localhost:8001"
echo "SLACK backend: http://localhost:8002"
echo
echo "Frontend UI:"
echo "Single UI instance: http://localhost:3000"
echo "(Use mode switcher to toggle between ON_CALL, FRIDAY, and SLACK backends)"
echo
echo "To start the UI, run: ./start_ui.sh"
echo
echo "Logs are being written to:"
echo "- logs/oncall_backend.log"
echo "- logs/friday_backend.log" 
echo "- logs/slack_backend.log"
echo
echo "To stop all servers, run: ./stop_servers.sh"
echo
echo "Servers are running in background. You can close this terminal." 