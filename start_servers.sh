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
echo "Waiting for servers to start up..."
sleep 5

# Function to check if server is responding
check_server_health() {
    local port=$1
    local name=$2
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/docs | grep -q "200\|404"; then
            echo "✓ $name is responding on port $port"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts: $name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "✗ $name failed to start on port $port"
    return 1
}

# Function to check if process is still running
check_process() {
    local pid=$1
    local name=$2
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "✓ $name process is running (PID: $pid)"
        return 0
    else
        echo "✗ $name process died (PID: $pid)"
        return 1
    fi
}

echo
echo "Verifying server status..."
echo

# Check processes
check_process $ONCALL_PID "ON_CALL backend"
check_process $FRIDAY_PID "FRIDAY backend"
check_process $SLACK_PID "SLACK backend"

echo
echo "Checking server responsiveness..."
echo

# Check server health
ONCALL_OK=false
FRIDAY_OK=false
SLACK_OK=false

if check_server_health 8000 "ON_CALL backend"; then
    ONCALL_OK=true
fi

if check_server_health 8001 "FRIDAY backend"; then
    FRIDAY_OK=true
fi

if check_server_health 8002 "SLACK backend"; then
    SLACK_OK=true
fi

echo
echo "========================================="
echo "SERVER STATUS SUMMARY"
echo "========================================="

if [ "$ONCALL_OK" = true ]; then
    echo "✓ ON_CALL backend: http://localhost:8000 - RUNNING"
else
    echo "✗ ON_CALL backend: http://localhost:8000 - FAILED"
fi

if [ "$FRIDAY_OK" = true ]; then
    echo "✓ FRIDAY backend: http://localhost:8001 - RUNNING"
else
    echo "✗ FRIDAY backend: http://localhost:8001 - FAILED"
fi

if [ "$SLACK_OK" = true ]; then
    echo "✓ SLACK backend: http://localhost:8002 - RUNNING"
else
    echo "✗ SLACK backend: http://localhost:8002 - FAILED"
fi

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

# Check if any servers failed
if [ "$ONCALL_OK" = false ] || [ "$FRIDAY_OK" = false ] || [ "$SLACK_OK" = false ]; then
    echo "⚠️  Some servers failed to start. Check the logs for details:"
    echo "   tail -f logs/*.log"
    echo
    echo "Common issues:"
    echo "- Make sure you're in the correct directory (project root)"
    echo "- Ensure virtual environment is activated"
    echo "- Check if AIQ toolkit is installed: aiq --version"
    echo "- Verify config files exist"
    exit 1
else
    echo "🎉 All servers are running successfully!"
fi

echo
echo "Servers are running in background. You can close this terminal." 