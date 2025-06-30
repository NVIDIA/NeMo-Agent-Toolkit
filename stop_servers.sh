#!/bin/bash

echo "Stopping AIQ Toolkit Backend Servers..."
echo

# Function to stop a server by PID file
stop_server() {
    local pid_file=$1
    local server_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $server_name (PID: $pid)..."
            kill $pid
            # Wait a moment for graceful shutdown
            sleep 2
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "Force stopping $server_name..."
                kill -9 $pid
            fi
            echo "$server_name stopped successfully"
        else
            echo "$server_name was not running (PID $pid not found)"
        fi
        rm -f "$pid_file"
    else
        echo "No PID file found for $server_name"
    fi
}

# Stop all servers
stop_server "logs/oncall_backend.pid" "ON_CALL backend"
stop_server "logs/friday_backend.pid" "FRIDAY backend" 
stop_server "logs/slack_backend.pid" "SLACK backend"

# Also try to kill any remaining aiq serve processes
echo
echo "Checking for any remaining aiq serve processes..."
pkill -f "aiq serve" 2>/dev/null && echo "Stopped additional aiq serve processes" || echo "No additional aiq serve processes found"

echo
echo "All backend servers have been stopped."
echo "You can now safely start them again with ./start_servers.sh" 