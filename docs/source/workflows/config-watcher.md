# Configuration File Watcher

The AIQ Toolkit configuration file watcher provides real-time monitoring of configuration files, enabling hot-reloading capabilities for improved development workflows. This feature allows developers to modify configuration files without needing to restart their applications.

## Overview

The configuration watcher consists of two main components:

1. **Event System** (`aiq.runtime.events`): Provides a centralized event management system for configuration changes
2. **File Watcher** (`aiq.runtime.config_watcher`): Monitors file system changes and dispatches events

## Quick Start

### Basic Usage

```python
from aiq.runtime.config_watcher import ConfigWatcher
from aiq.runtime.events import get_event_manager, ConfigEventType

# Create a watcher instance
watcher = ConfigWatcher()

# Register an event handler
def on_config_change(event):
    print(f"Configuration file {event.file_path} was {event.event_type}")

get_event_manager().register_handler(on_config_change)

# Add files to watch
watcher.add_file("config.yml")
watcher.add_file("overrides.yml")

# Start watching
with watcher:
    # Your application logic here
    # The watcher will automatically detect changes
    pass
```

### Event Types

The watcher detects the following types of configuration changes:

- `FILE_MODIFIED`: A watched file was modified
- `FILE_CREATED`: A watched file was created
- `FILE_DELETED`: A watched file was deleted
- `FILE_MOVED`: A watched file was moved/renamed

## Event System

### ConfigChangeEvent

Each configuration change generates a `ConfigChangeEvent` with the following properties:

```python
class ConfigChangeEvent:
    event_type: ConfigEventType      # Type of change (modified, created, etc.)
    file_path: Path                  # Path to the changed file
    timestamp: datetime              # When the change occurred
    old_path: Optional[Path]         # Previous path (for move events)
    checksum: Optional[str]          # File checksum for change detection
```

### Event Handlers

You can register event handlers for specific event types or all events:

```python
from aiq.runtime.events import get_event_manager, ConfigEventType

# Register handler for specific event type
def on_file_modified(event):
    print(f"File modified: {event.file_path}")

get_event_manager().register_handler(on_file_modified, ConfigEventType.FILE_MODIFIED)

# Register global handler (receives all events)
def on_any_change(event):
    print(f"Any change: {event.event_type} - {event.file_path}")

get_event_manager().register_handler(on_any_change)
```

### Recent Events

The event manager keeps track of recent events for debugging and monitoring:

```python
from aiq.runtime.events import get_event_manager

# Get recent events (most recent first)
recent_events = get_event_manager().get_recent_events(limit=10)

for event in recent_events:
    print(f"{event.timestamp}: {event.event_type} - {event.file_path}")
```

## File Watcher

### ConfigWatcher Class

The `ConfigWatcher` class provides the main interface for monitoring configuration files:

```python
from aiq.runtime.config_watcher import ConfigWatcher

# Create watcher with custom debounce delay
watcher = ConfigWatcher(debounce_delay=0.2)

# Add files to watch
watcher.add_file("config.yml")
watcher.add_file("database.yml")

# Start/stop watching
watcher.start()
# ... application logic ...
watcher.stop()

# Or use as context manager
with watcher:
    # ... application logic ...
    pass
```

### Adding and Removing Files

```python
# Add a file to watch
watcher.add_file("/path/to/config.yml")

# Remove a file from watching
watcher.remove_file("/path/to/config.yml")

# Get list of watched files
watched_files = watcher.get_watched_files()
```

### Debouncing

The watcher includes built-in debouncing to prevent excessive events from rapid file changes:

```python
# Create watcher with 0.5 second debounce delay
watcher = ConfigWatcher(debounce_delay=0.5)
```

When a file is modified multiple times within the debounce period, only one event is generated.

## Advanced Usage

### Custom Event Processing

```python
from aiq.runtime.events import get_event_manager, ConfigEventType
from aiq.runtime.config_watcher import ConfigWatcher
import logging

logger = logging.getLogger(__name__)

class ConfigReloader:
    def __init__(self):
        self.watcher = ConfigWatcher()
        self.current_config = None

        # Register event handlers
        get_event_manager().register_handler(
            self.on_config_modified,
            ConfigEventType.FILE_MODIFIED
        )
        get_event_manager().register_handler(
            self.on_config_deleted,
            ConfigEventType.FILE_DELETED
        )

    def on_config_modified(self, event):
        """Handle configuration file modifications."""
        try:
            logger.info(f"Reloading configuration from {event.file_path}")
            self.reload_config(event.file_path)
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")

    def on_config_deleted(self, event):
        """Handle configuration file deletions."""
        logger.warning(f"Configuration file deleted: {event.file_path}")
        # Handle deletion logic here

    def reload_config(self, config_path):
        """Reload configuration from file."""
        # Your configuration loading logic here
        pass

    def start_watching(self, config_file):
        """Start watching a configuration file."""
        self.watcher.add_file(config_file)
        self.watcher.start()

    def stop_watching(self):
        """Stop watching configuration files."""
        self.watcher.stop()
```

### Multiple File Handling

```python
from pathlib import Path
from aiq.runtime.config_watcher import ConfigWatcher

# Watch multiple configuration files
config_files = [
    Path("config/app.yml"),
    Path("config/database.yml"),
    Path("config/logging.yml")
]

watcher = ConfigWatcher()

for config_file in config_files:
    if config_file.exists():
        watcher.add_file(config_file)

with watcher:
    # All files are now being monitored
    pass
```

### Error Handling

```python
from aiq.runtime.events import get_event_manager
import logging

logger = logging.getLogger(__name__)

def robust_config_handler(event):
    """Configuration handler with error handling."""
    try:
        # Process the configuration change
        process_config_change(event)
    except Exception as e:
        logger.error(f"Error processing config change: {e}", exc_info=True)
        # Optionally implement fallback logic
        handle_config_error(event, e)

def process_config_change(event):
    """Process configuration change."""
    # Your configuration processing logic
    pass

def handle_config_error(event, error):
    """Handle configuration processing errors."""
    # Your error handling logic
    pass

# Register the robust handler
get_event_manager().register_handler(robust_config_handler)
```

## Integration with AIQ Toolkit

The configuration watcher is designed to integrate seamlessly with the AIQ Toolkit workflow system. Future versions will include:

- Automatic integration with `aiq run` and `aiq serve` commands
- Development mode (`--dev` flag) for automatic reloading
- Configuration validation before applying changes
- Rollback capabilities for invalid configurations

## Testing

### Unit Testing

When testing code that uses the configuration watcher, use the reset function to ensure clean state:

```python
from aiq.runtime.events import reset_event_manager
import pytest

class TestMyConfigHandler:
    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()

    def teardown_method(self):
        """Clean up test fixtures."""
        reset_event_manager()

    def test_config_handling(self):
        """Test configuration event handling."""
        # Your test code here
        pass
```

### Integration Testing

For integration tests, you can simulate file changes and verify that events are processed correctly:

```python
import tempfile
import time
from pathlib import Path
from aiq.runtime.config_watcher import ConfigWatcher
from aiq.runtime.events import get_event_manager

def test_file_change_detection():
    """Test that file changes are detected."""
    events_received = []

    def event_handler(event):
        events_received.append(event)

    get_event_manager().register_handler(event_handler)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("test: config\n")
        temp_file = Path(f.name)

    try:
        watcher = ConfigWatcher()
        watcher.add_file(temp_file)

        with watcher:
            # Modify the file
            temp_file.write_text("test: modified\n")

            # Wait for event processing
            time.sleep(0.2)

        # Verify event was received
        assert len(events_received) > 0
        assert events_received[0].file_path == temp_file

    finally:
        temp_file.unlink()  # Clean up
```

## Performance Considerations

### Debouncing

The default debounce delay is 0.1 seconds, which provides a good balance between responsiveness and performance. For high-frequency changes, consider increasing the delay:

```python
# Increase debounce delay for better performance
watcher = ConfigWatcher(debounce_delay=0.5)
```

### File Checksums

The watcher uses SHA256 checksums to detect actual file changes and avoid processing false positives from file system events. This adds minimal overhead while ensuring accuracy.

### Resource Usage

- The watcher creates one thread per monitored directory
- File checksums are calculated only when files change
- Event history is limited to 100 recent events by default

## Troubleshooting

### Common Issues

1. **Events not firing**: Ensure the watcher is started and the file exists
2. **Multiple events for one change**: Check debounce delay settings
3. **High CPU usage**: Verify debounce delay is appropriate for your use case

### Debug Logging

Enable debug logging to troubleshoot watcher behavior:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aiq.runtime.config_watcher')
logger.setLevel(logging.DEBUG)
```

### Event History

Use the event manager's recent events feature to debug event processing:

```python
from aiq.runtime.events import get_event_manager

# Get recent events for debugging
recent_events = get_event_manager().get_recent_events(limit=20)

for event in recent_events:
    print(f"{event.timestamp}: {event.event_type} - {event.file_path}")
```

## Next Steps

This configuration watcher provides the foundation for hot-reloading functionality in the AIQ Toolkit. The next development phases will include:

1. **Configuration Hot-Reload**: Automatic configuration reloading with validation
2. **CLI Development Mode**: Integration with `aiq run` and `aiq serve` commands
3. **Plugin Hot-Reload**: Dynamic plugin loading and unloading

For more information about these upcoming features, see the [Hot-Reload Roadmap](hot-reload-roadmap.md).
