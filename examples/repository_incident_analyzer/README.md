# Repository & Incident Analyzer

A comprehensive analysis tool that combines repository analysis (GitLab/GitHub) with incident management (PagerDuty) capabilities using the AIQ Toolkit.

## Features

### Repository Analysis
- **GitLab Support**: Read files from GitLab repositories (both public and private)
- **GitHub Support**: Read files from GitHub repositories with line-specific reading
- **Comprehensive Analysis**: Analyzes code structure, dependencies, architecture, and documentation
- **Multi-branch Support**: Automatically tries different branches (main, master, develop) if files are not found

### Incident Analysis
- **PagerDuty Integration**: Fetch and analyze incident details from PagerDuty URLs
- **Incident Timeline**: Analyze incident creation, updates, and resolution
- **Service Impact**: Assess affected services and business impact
- **Response Team**: Review assigned responders and escalation paths

## Installation

1. Install the package in development mode:
```bash
cd examples/repository_incident_analyzer
pip install -e .
```

2. Set up environment variables:
```bash
# For GitLab private repositories
export GITLAB_API_TOKEN="your_gitlab_token_here"

# For PagerDuty incidents
export PAGERDUTY_API_TOKEN="your_pagerduty_token_here"
```

## Configuration Files

The project includes three configuration files:

### 1. Combined Analyzer (`combined_analyzer.yml`)
Includes both repository and incident analysis capabilities. Use this for comprehensive DevOps analysis.

### 2. Repository Only (`repository_only.yml`) 
Focus exclusively on GitLab/GitHub repository analysis.

### 3. Incident Only (`incident_only.yml`)
Focus exclusively on PagerDuty incident analysis.

## Usage

### Running the Combined Analyzer
```bash
aiq workflow run --config-path configs/combined_analyzer.yml
```

### Running Repository Analysis Only
```bash
aiq workflow run --config-path configs/repository_only.yml
```

### Running Incident Analysis Only
```bash
aiq workflow run --config-path configs/incident_only.yml
```

## Example Queries

### Repository Analysis
- "Analyze this GitLab repository: https://gitlab.com/group/project/-/blob/main/README.md"
- "Compare the architecture of these GitHub repositories: [URLs]"
- "What are the dependencies and setup requirements for this project?"

### Incident Analysis
- "Analyze this PagerDuty incident: https://your-domain.pagerduty.com/incidents/INCIDENT_ID"
- "What was the root cause and impact of this incident?"
- "How can we prevent similar incidents in the future?"

### Combined Analysis
- "Analyze this repository and any related incidents from our PagerDuty"
- "Is there a correlation between this code change and recent incidents?"

## API Tokens

### GitLab API Token
1. Go to GitLab → User Settings → Access Tokens
2. Create a personal access token with `read_repository` scope
3. Set the `GITLAB_API_TOKEN` environment variable

### PagerDuty API Token
1. Go to PagerDuty → Integrations → API Access Keys
2. Create a new API key with read permissions
3. Set the `PAGERDUTY_API_TOKEN` environment variable

## Architecture

```
repository_incident_analyzer/
├── src/repository_incident_analyzer/
│   ├── __init__.py              # Package initialization
│   ├── gitlab_tools.py          # GitLab API integration
│   ├── pagerduty_tools.py       # PagerDuty API integration
│   └── register.py              # Tool registration
├── configs/
│   ├── combined_analyzer.yml    # Combined analysis config
│   ├── repository_only.yml      # Repository-only config
│   └── incident_only.yml        # Incident-only config
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

## Supported URL Formats

### GitLab URLs
```
https://gitlab.com/group/project/-/blob/branch/path/file.ext
https://gitlab-instance.com/group/project/-/blob/branch/path/file.ext
```

### GitHub URLs
```
https://github.com/owner/repo/blob/branch/path/file.ext
```

### PagerDuty URLs
```
https://your-domain.pagerduty.com/incidents/INCIDENT_ID
```

## Contributing

This project combines functionality from two separate AIQ Toolkit examples:
- `gitlab_github_analyzer`: Repository analysis capabilities
- `pagerduty_tool`: Incident management capabilities

When contributing, please ensure that changes maintain compatibility with both analysis types.

## License

SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
SPDX-License-Identifier: Apache-2.0 