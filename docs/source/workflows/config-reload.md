# Configuration Hot-Reload

The AIQ Toolkit configuration hot-reload functionality allows developers to reload configuration files during development without restarting their applications. This feature builds upon the configuration file watcher infrastructure to provide a complete development workflow.

## Overview

Configuration hot-reload consists of three main components:

1. **ConfigManager** (`aiq.runtime.config_manager`): Manages configuration lifecycle with reload, validation, and rollback capabilities
2. **Enhanced Loader** (`aiq.runtime.loader`): Provides simple reload functions and config manager creation
3. **Manual Trigger** (`aiq.runtime.manual_reload_trigger`): Interactive CLI tool for testing and development

## Quick Start

### Simple Configuration Reload

```python
from aiq.runtime.loader import reload_config

# Reload configuration from file
new_config = reload_config("config.yml")

# Or validate only without applying changes
validated_config = reload_config("config.yml", validate_only=True)
```

### Advanced Configuration Management

```python
from aiq.runtime.loader import create_config_manager

# Create a configuration manager
with create_config_manager("config.yml") as manager:
    # Set configuration overrides
    manager.set_overrides({
        "llms.temperature": "0.7",
        "workflows.timeout": "60"
    })

    # Reload configuration
    new_config = manager.reload_config()

    # Rollback if needed
    if something_went_wrong:
        manager.rollback_config()
```

## Configuration Manager

### ConfigManager Class

The `ConfigManager` class provides comprehensive configuration lifecycle management:

```python
from aiq.runtime.config_manager import ConfigManager

with ConfigManager("config.yml") as manager:
    # Get current configuration
    current_config = manager.current_config

    # Check reload count
    print(f"Configuration has been reloaded {manager.reload_count} times")

    # Reload configuration
    try:
        new_config = manager.reload_config()
        print("Configuration reloaded successfully!")
    except ConfigValidationError as e:
        print(f"Configuration validation failed: {e}")
    except ConfigReloadError as e:
        print(f"Configuration reload failed: {e}")
```

### Configuration Overrides

Preserve configuration overrides across reloads:

```python
# Set overrides that will be preserved during reloads
overrides = {
    "llms.nim_llm.temperature": "0.8",
    "tools.search.timeout": "30",
    "workflows.main.max_iterations": "10"
}

with ConfigManager("config.yml") as manager:
    manager.set_overrides(overrides)

    # Overrides are automatically reapplied after each reload
    manager.reload_config()

    # Check current overrides
    current_overrides = manager.config_overrides
    print(f"Active overrides: {len(current_overrides)}")
```

### Configuration Snapshots and Rollback

The configuration manager automatically creates snapshots for rollback:

```python
with ConfigManager("config.yml") as manager:
    # View snapshots
    snapshots = manager.get_snapshots()
    print(f"Available snapshots: {len(snapshots)}")

    for i, snapshot in enumerate(snapshots):
        print(f"  {i+1}. {snapshot.timestamp}")

    # Make some changes
    manager.reload_config()  # Creates snapshot before reload
    manager.reload_config()  # Another snapshot

    # Rollback to previous version
    try:
        rolled_back_config = manager.rollback_config(steps=1)
        print("Rolled back to previous configuration")
    except ConfigReloadError as e:
        print(f"Rollback failed: {e}")

    # Clear old snapshots (keeps current)
    manager.clear_snapshots()
```

## Manual Reload Trigger

For development and testing, use the manual reload trigger CLI:

### Simple Reload

```bash
# Reload configuration
python -m aiq.runtime.manual_reload_trigger -c config.yml reload

# Validate configuration without applying changes
python -m aiq.runtime.manual_reload_trigger -c config.yml reload --validate-only

# Enable verbose logging
python -m aiq.runtime.manual_reload_trigger -c config.yml -v reload
```

### Interactive Session

```bash
# Start interactive session
python -m aiq.runtime.manual_reload_trigger -c config.yml interactive

# Start with configuration overrides
python -m aiq.runtime.manual_reload_trigger -c config.yml interactive \
  -o llms.temperature=0.8 \
  -o workflows.timeout=60
```

#### Interactive Commands

Once in interactive mode, use these commands:

- **r** - Reload configuration
- **v** - Validate configuration
- **s** - Show snapshots
- **b [steps]** - Rollback configuration (default: 1 step)
- **c** - Clear snapshots
- **i** - Show current config info
- **q** - Quit

### Example Interactive Session

```
🚀 Starting interactive configuration manager for config.yml
📍 Interactive session started. Available commands:
   r - Reload configuration
   v - Validate configuration
   s - Show snapshots
   b [steps] - Rollback configuration
   c - Clear snapshots
   i - Show current config info
   q - Quit

Command: r
🔄 Reloading configuration...
✅ Configuration reloaded successfully (reload #1)
📋 Configuration summary:
   - LLMs: 2
   - Tools: 5
   - Workflows: 1

Command: s
📸 Configuration snapshots (2 total):
📍 1. 2025-01-15 10:30:45
   2. 2025-01-15 10:25:12

Command: b
⏪ Rolling back 1 step...
✅ Rollback completed (reload #2)
📋 Configuration summary:
   - LLMs: 2
   - Tools: 5
   - Workflows: 1

Command: q
👋 Goodbye!
```

## Error Handling

### Validation Errors

When configuration validation fails, the reload is aborted and previous configuration is preserved:

```python
from aiq.runtime.config_manager import ConfigValidationError

try:
    manager.reload_config()
except ConfigValidationError as e:
    print(f"Configuration validation failed: {e}")
    # Previous configuration is still active
    # Use rollback if needed
    manager.rollback_config()
```

### Reload Errors

Handle general reload errors:

```python
from aiq.runtime.config_manager import ConfigReloadError

try:
    manager.reload_config()
except ConfigReloadError as e:
    print(f"Configuration reload failed: {e}")
    # Check if file exists, permissions, etc.
```

### Rollback Errors

Handle rollback limitations:

```python
try:
    manager.rollback_config(steps=5)
except ConfigReloadError as e:
    print(f"Cannot rollback 5 steps: {e}")

    # Check available snapshots
    snapshots = manager.get_snapshots()
    max_rollback = len(snapshots) - 1
    print(f"Maximum rollback steps: {max_rollback}")
```

## Integration with File Watcher

Combine configuration reload with file watching for development:

```python
from aiq.runtime.config_watcher import ConfigWatcher
from aiq.runtime.config_manager import ConfigManager
from aiq.runtime.events import get_event_manager, ConfigEventType

# Set up file change handler
def on_config_change(event):
    print(f"Configuration file changed: {event.file_path}")
    # Manual trigger - automatic reload in Step 3

get_event_manager().register_handler(on_config_change, ConfigEventType.FILE_MODIFIED)

# Start watching and managing configuration
with ConfigWatcher() as watcher:
    watcher.add_file("config.yml")

    with ConfigManager("config.yml") as manager:
        # Development workflow
        while True:
            user_input = input("Press 'r' to reload, 'q' to quit: ")
            if user_input == 'r':
                try:
                    manager.reload_config()
                    print("✅ Configuration reloaded!")
                except Exception as e:
                    print(f"❌ Reload failed: {e}")
            elif user_input == 'q':
                break
```

## Best Practices

### Development Workflow

1. **Use validation first**: Always validate configuration before applying changes
2. **Keep snapshots**: Don't clear snapshots frequently during active development
3. **Monitor overrides**: Check that overrides are properly reapplied after reload
4. **Test rollback**: Verify rollback functionality works with your configuration structure

### Configuration Design

1. **Validate early**: Ensure your configuration schema catches errors quickly
2. **Modular structure**: Design configurations that can be partially reloaded
3. **Override-friendly**: Structure configurations to work well with dot-notation overrides
4. **Backward compatibility**: Consider how configuration changes affect existing snapshots

### Error Recovery

1. **Graceful degradation**: Handle reload failures without crashing applications
2. **Clear error messages**: Provide helpful error messages for configuration issues
3. **Rollback strategy**: Have a clear rollback strategy for failed reloads
4. **Logging**: Enable appropriate logging for debugging reload issues

## Configuration Examples

### Basic Configuration

```yaml
# config.yml
llms:
  main_llm:
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 1000

tools:
  search:
    timeout: 30
    max_results: 10

workflows:
  main:
    name: "Main Workflow"
    max_iterations: 5
```

### Configuration with Overrides

```python
# Apply overrides for development
overrides = {
    "llms.main_llm.temperature": "0.9",  # Higher temperature for creativity
    "tools.search.timeout": "60",        # Longer timeout for debugging
    "workflows.main.max_iterations": "10" # More iterations for testing
}

with ConfigManager("config.yml") as manager:
    manager.set_overrides(overrides)

    # Configuration now uses override values
    config = manager.current_config
```

### Multiple Environment Configurations

```python
# Load different configurations based on environment
import os

env = os.getenv("ENV", "development")
config_file = f"config-{env}.yml"

with ConfigManager(config_file) as manager:
    if env == "development":
        # Development-specific overrides
        manager.set_overrides({
            "llms.main_llm.temperature": "0.9",
            "tools.search.timeout": "60"
        })

    # Use configuration for current environment
    config = manager.current_config
```

## Troubleshooting

### Common Issues

1. **Configuration not reloading**: Check file permissions and existence
2. **Overrides not applied**: Verify override paths match configuration structure
3. **Validation failures**: Check YAML syntax and schema compliance
4. **Rollback fails**: Ensure sufficient snapshots exist

### Debug Logging

Enable debug logging to troubleshoot reload issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aiq.runtime.config_manager')
logger.setLevel(logging.DEBUG)

# Now reload operations will show detailed logging
manager.reload_config()
```

### Performance Considerations

- Configuration reloading is designed to be fast, but large configurations may take time to validate
- Snapshots use memory proportional to configuration size
- Override reapplication is O(n) where n is the number of overrides

## Next Steps

This configuration hot-reload functionality provides the foundation for automatic reloading. The next development phase will include:

1. **CLI Development Mode**: Integration with `aiq run` and `aiq serve` commands with `--dev` flag
2. **Automatic Reloading**: Trigger reloads automatically when files change
3. **Advanced Validation**: Schema-aware validation with better error messages

For information about upcoming automatic reload features, see the [Development Mode Guide](development-mode.md).
