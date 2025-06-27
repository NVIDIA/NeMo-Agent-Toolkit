# Dual-Mode Configuration System

This example demonstrates how to run AIQ Toolkit with separate configurations for FRIDAY and ON CALL modes, allowing different tools, prompts, and behavior for each mode.

## Overview

The dual-mode system allows you to:
- Run different YAML configuration files for each mode
- Have mode-specific tools, prompts, and LLM settings
- Switch between modes in the UI while maintaining separate chat histories
- Use different workflows optimized for different use cases

## Configuration Files

### FRIDAY Mode (`friday_analyzer.yml`)
- **Purpose**: General repository analysis and development assistance
- **Temperature**: 0.7 (more creative responses)
- **Max Tokens**: 2048
- **Tools**: `github_analyzer` for general repository analysis
- **Focus**: Code review, documentation, best practices

### ON CALL Mode (`on_call_analyzer.yml`)
- **Purpose**: Incident response and critical issue analysis
- **Temperature**: 0.1 (more focused, deterministic responses)
- **Max Tokens**: 4096 (longer analysis for incidents)
- **Tools**: `incident_analyzer`, `pagerduty_analyzer`
- **Focus**: Critical bugs, performance issues, security, emergency response

## Running with Dual-Mode Configuration

### Option 1: Separate Config Files
Run with different configurations for each mode:

```bash
aiq serve --friday_config_file=examples/repository_incident_analyzer/configs/friday_analyzer.yml \
          --on_call_config_file=examples/repository_incident_analyzer/configs/on_call_analyzer.yml
```

### Option 2: Single Config for Both Modes
Use the same configuration for both modes:

```bash
aiq serve --config_file=examples/repository_incident_analyzer/configs/combined_analyzer.yml
```

### Option 3: Mixed Configuration
Use one config file for both modes, but specify a different one for a specific mode:

```bash
# Use friday_analyzer.yml for both modes
aiq serve --friday_config_file=examples/repository_incident_analyzer/configs/friday_analyzer.yml

# Use on_call_analyzer.yml for ON CALL mode, friday_analyzer.yml for FRIDAY mode
aiq serve --friday_config_file=examples/repository_incident_analyzer/configs/friday_analyzer.yml \
          --on_call_config_file=examples/repository_incident_analyzer/configs/on_call_analyzer.yml
```

## UI Mode Switching

Once the server is running, you can:

1. **Access the UI**: Navigate to `http://localhost:8000`
2. **Switch Modes**: Use the mode switcher in the sidebar to toggle between FRIDAY and ON CALL
3. **Separate Conversations**: Each mode maintains its own chat history
4. **Different Behavior**: Notice how the AI responds differently in each mode based on the configuration

## Mode-Specific Features

### FRIDAY Mode Features
- General repository analysis questions
- Code review assistance
- Documentation help
- Best practices recommendations
- More creative and detailed responses

### ON CALL Mode Features
- Incident response queries
- Critical bug analysis
- Performance troubleshooting
- Security assessment
- PagerDuty alert correlation
- Focused, actionable responses

## Example Interactions

### FRIDAY Mode
```
User: "How can I improve the code quality in this repository?"
FRIDAY: [Provides detailed code review suggestions, documentation improvements, best practices]
```

### ON CALL Mode
```
User: "We have a critical performance issue in production. Here's the PagerDuty alert: [link]"
ON CALL: [Analyzes the alert, correlates with repository issues, provides immediate mitigation steps]
```

## Environment Variables

Make sure to set the required environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Architecture

The dual-mode system works by:

1. **CLI Enhancement**: Extended `aiq serve` to accept `--friday_config_file` and `--on_call_config_file` parameters
2. **Backend Mode Manager**: Created `ModeConfigManager` to handle mode-specific configurations
3. **Session Management**: Modified WebSocket and HTTP handlers to use mode-specific session managers
4. **Frontend Integration**: Updated UI to send mode information with requests
5. **Separate Workflows**: Each mode uses its own workflow builder and configuration

This allows for completely different AI behavior and capabilities based on the selected mode, while maintaining a unified user interface. 