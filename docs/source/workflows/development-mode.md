# Development Mode

AIQ Toolkit provides a powerful development mode that enables automatic configuration reloading during development. This eliminates the need to manually restart applications when making configuration changes, significantly improving the development workflow.

## Overview

Development mode provides:

- **Automatic Configuration Reloading**: Detects changes to configuration files and automatically reloads them
- **Hot-Swapping**: Updates running workflows without full application restart
- **Error Handling**: Safe reloading with automatic rollback on validation failures
- **Override Preservation**: Maintains command-line overrides across reloads
- **Real-time Feedback**: Visual indicators and logging for reload operations

## Enabling Development Mode

Add the `--dev` flag to any `aiq run` or `aiq serve` command:

```bash
# Enable development mode for console workflows
aiq run console --config_file config.yml --dev

# Enable development mode for FastAPI server
aiq serve fastapi --config_file config.yml --dev

# With configuration overrides
aiq run console --config_file config.yml --dev --override llms.openai.temperature 0.8
```

## How It Works

When development mode is enabled:

1. **File Watching**: Monitors the configuration file for changes using efficient file system events
2. **Change Detection**: Detects file modifications, creations, and moves with intelligent debouncing
3. **Validation**: Validates new configuration before applying changes
4. **Hot Reload**: Updates the running workflow with the new configuration
5. **Rollback**: Automatically rolls back to the previous configuration if validation fails

## Visual Feedback

Development mode provides rich visual feedback:

```
Development Mode Active
==================================================
Watching config: /path/to/config.yml
Reload delay: 0.5s

Configuration changes will be automatically reloaded
Press Ctrl+C to stop development mode
==================================================

Starting AIQ Toolkit from config file: '/path/to/config.yml'
Development mode enabled - automatic reloading active

Configuration file changed: /path/to/config.yml
Reloading configuration...
Configuration reloaded successfully
Starting workflow with updated configuration...
```

## Configuration Examples

### Basic Development Workflow

1. **Start with development mode**:
   ```bash
   aiq run console --config_file workflow.yml --dev
   ```

2. **Edit your configuration file** (workflow.yml):
   ```yaml
   general:
     front_end:
       _type: console

   llms:
     openai_llm:
       _type: openai
       temperature: 0.7  # Change this value
   ```

3. **Save the file** - The workflow automatically reloads with new settings

### FastAPI Server Development

For web applications, development mode is particularly useful:

```bash
aiq serve fastapi --config_file api_config.yml --dev --override general.front_end.port 8001
```

Update the configuration while the server is running:

```yaml
general:
  front_end:
    _type: fastapi
    port: 8000
    host: "0.0.0.0"

llms:
  nim_llm:
    _type: nim
    model: "meta/llama-3.1-8b-instruct"
    base_url: "http://localhost:8080/v1"
    temperature: 0.9  # Hot-reload this change!
```

### Complex Configuration Changes

Development mode handles complex configuration changes:

```yaml
# Add new LLM provider
llms:
  existing_llm:
    _type: openai
    temperature: 0.7

  new_llm:  # This gets hot-loaded
    _type: anthropic
    model: "claude-3-sonnet"
    temperature: 0.8

# Update tools configuration
tools:
  web_search:
    _type: tavily
    max_results: 10  # Increase from 5
```

## Error Handling and Rollback

Development mode includes robust error handling:

### Validation Errors

If a configuration change is invalid:

```
Configuration file changed: /path/to/config.yml
Reloading configuration...
Configuration reload failed (attempt 1/3): Validation error: Unknown LLM type 'invalid_llm'
Workflow continues with previous configuration due to reload error
```

### Automatic Rollback

After multiple failed attempts:

```
Configuration reload failed (attempt 3/3): Validation error
Maximum reload attempts exceeded, attempting rollback...
Configuration rolled back successfully
Starting workflow with updated configuration...
```

## Advanced Features

### Configuration Overrides

Command-line overrides are preserved across reloads:

```bash
aiq run console --config_file config.yml --dev \
  --override llms.openai.temperature 0.9 \
  --override tools.web_search.max_results 15
```

Even when the configuration file changes, these overrides remain active.

### Multiple File Watching

Development mode can watch additional files (implementation detail for future extension):

```python
# Future API for watching multiple files
dev_manager = DevModeManager(
    config_file="config.yml",
    watch_additional_files={
        Path("prompts.yml"),
        Path("tools_config.yml")
    }
)
```

## Best Practices

### Configuration Structure

Organize your configuration for easy development:

```yaml
# workflow.yml
general:
  front_end:
    _type: console

# Separate development and production configs
llms: !include llms_dev.yml
tools: !include tools_dev.yml
```

### Development vs. Production

Use different configurations for development and production:

```bash
# Development
aiq run console --config_file config_dev.yml --dev

# Production
aiq run console --config_file config_prod.yml
```

### Testing Configuration Changes

1. **Start with working configuration**
2. **Make incremental changes**
3. **Test each change immediately**
4. **Use version control** to track working configurations

### Performance Considerations

- **File System Events**: Development mode uses efficient file system events, not polling
- **Debouncing**: Multiple rapid changes are debounced to avoid excessive reloads
- **Validation Caching**: Configuration validation is optimized for repeated use
- **Memory Management**: Old configurations are cleaned up automatically

## Troubleshooting

### Common Issues

**Issue**: Configuration not reloading
```bash
# Check file permissions
ls -la config.yml

# Ensure file exists and is writable
touch config.yml
```

**Issue**: Validation errors persist
```bash
# Validate configuration manually
aiq validate --config_file config.yml
```

**Issue**: Override not working
```bash
# Check override syntax
aiq run console --config_file config.yml --dev --override llms.openai.temperature 0.8
```

### Debug Mode

Enable debug logging for detailed information:

```bash
export AIQ_LOG_LEVEL=DEBUG
aiq run console --config_file config.yml --dev
```

### Manual Testing

Test configuration reloading manually using the manual reload trigger:

```bash
# Test reload without development mode
python -m aiq.runtime.manual_reload_trigger -c config.yml reload

# Interactive testing
python -m aiq.runtime.manual_reload_trigger -c config.yml interactive
```

## API Reference

### CLI Options

- `--dev`: Enable development mode with automatic reloading
- `--config_file`: Configuration file to watch and reload (required)
- `--override`: Configuration overrides (preserved across reloads)

### Environment Variables

- `AIQ_LOG_LEVEL`: Set to `DEBUG` for detailed reload logging
- `AIQ_DEV_MODE_DELAY`: Override default reload delay (default: 0.5 seconds)

### Signals

Development mode responds to standard signals:

- `SIGINT` (Ctrl+C): Graceful shutdown
- `SIGTERM`: Graceful shutdown

## Integration Examples

### CI/CD Pipeline

```bash
# Test configuration reloading in CI
aiq validate --config_file config.yml
aiq run console --config_file config.yml --dev &
PID=$!

# Make test changes
echo "# Test change" >> config.yml
sleep 2

# Clean up
kill $PID
```

### Docker Development

```dockerfile
# Dockerfile for development
FROM aiq-base:latest

# Enable development mode
CMD ["aiq", "serve", "fastapi", "--config_file", "/app/config.yml", "--dev"]

# Mount configuration directory
VOLUME ["/app"]
```

### IDE Integration

Most IDEs work seamlessly with development mode:

- **VS Code**: File changes trigger automatic reloads
- **PyCharm**: Works with auto-save features
- **Vim/Emacs**: Manual saves trigger reloads

## Security Considerations

Development mode is designed for development environments:

- **File System Access**: Requires read access to configuration files
- **Network Binding**: FastAPI server may bind to all interfaces
- **Debug Information**: Additional logging may expose configuration details

**Never use development mode in production environments.**

## Related Features

- [Configuration Management](config-reload.md): Manual configuration reloading
- [File Watching](config-watcher.md): Low-level file monitoring infrastructure
- [CLI Reference](../reference/cli.md): Complete CLI documentation
- [Configuration Reference](../reference/configuration.md): Configuration file format

## Migration Guide

### From Manual Restarts

**Before**: Manual application restarts
```bash
# Old workflow
aiq run console --config_file config.yml
# Edit config.yml
# Ctrl+C to stop
aiq run console --config_file config.yml  # Restart
```

**After**: Development mode
```bash
# New workflow
aiq run console --config_file config.yml --dev
# Edit config.yml - automatic reload!
```

### From External Tools

**Before**: Using external file watchers
```bash
# Old approach with entr
ls config.yml | entr -r aiq run console --config_file config.yml
```

**After**: Built-in development mode
```bash
# New approach
aiq run console --config_file config.yml --dev
```

## Future Enhancements

Planned improvements for development mode:

- **Multi-file Watching**: Monitor multiple configuration files
- **Plugin Hot-reloading**: Reload plugins without restart
- **Interactive Configuration**: Live configuration editing interface
- **Performance Profiling**: Monitor reload performance
- **Configuration Diffing**: Show what changed between reloads
