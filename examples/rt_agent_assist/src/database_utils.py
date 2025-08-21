import sqlite3
import json
from .data_models import SessionState, FormSubmission


class SessionStateDB:
    """Handles database operations for session state management"""
    
    def __init__(self, db_path: str = "session_state.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Transcripts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Detected events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detected_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Form submissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS form_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    status TEXT DEFAULT 'ready',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def get_session_state(self, session_id: str) -> SessionState:
        """Retrieve the current session state for a given session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get transcript
            cursor.execute("""
                SELECT content FROM transcripts 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            transcript_rows = cursor.fetchall()
            transcript = [row[0] for row in transcript_rows]
            
            # Get detected events
            cursor.execute("""
                SELECT event_type, event_data FROM detected_events 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            event_rows = cursor.fetchall()
            detected_events = []
            for event_type, event_data in event_rows:
                detected_events.append({
                    "type": event_type,
                    "data": json.loads(event_data)
                })
            
            # Get latest form submission
            cursor.execute("""
                SELECT function_name, data, status, timestamp FROM form_submissions 
                WHERE session_id = ? AND status = 'ready'
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (session_id,))
            form_row = cursor.fetchone()
            
            latest_form_submission = None
            if form_row:
                function_name, data, status, timestamp = form_row
                latest_form_submission = FormSubmission(
                    function_name=function_name,
                    data=json.loads(data),
                    status=status,
                    timestamp=timestamp
                )
            
            return SessionState(
                transcript=transcript,
                detected_events=detected_events,
                latest_form_submission=latest_form_submission
            )
    
    def add_detected_events(self, session_id: str, events: list[dict[str, Any]]):
        """Add new detected events to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for event in events:
                cursor.execute("""
                    INSERT INTO detected_events (session_id, event_type, event_data)
                    VALUES (?, ?, ?)
                """, (session_id, event["type"], json.dumps(event["data"])))
            
            conn.commit()
    
    def update_form_submission_status(self, session_id: str, function_name: str, status: str):
        """Update the status of a form submission"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE form_submissions 
                SET status = ? 
                WHERE session_id = ? AND function_name = ? AND status = 'ready'
            """, (status, session_id, function_name))
            
            conn.commit()
    
    def get_previously_detected_intents(self, session_id: str) -> dict[str, list[str]]:
        """Get lists of previously detected function and data intents"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT event_type, event_data FROM detected_events 
                WHERE session_id = ? AND event_type IN ('function_intent', 'data_intent')
                ORDER BY timestamp ASC
            """, (session_id,))
            
            rows = cursor.fetchall()
            
            function_intents = []
            data_intents = []
            
            for event_type, event_data_str in rows:
                event_data = json.loads(event_data_str)
                if event_type == "function_intent":
                    function_intents.append(event_data.get("name", ""))
                elif event_type == "data_intent":
                    data_intents.append(event_data.get("name", ""))
            
            return {
                "function_intents": function_intents,
                "data_intents": data_intents
            }
