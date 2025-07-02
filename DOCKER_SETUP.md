# Docker Setup Guide for NeMo Agent Toolkit

This guide explains how to run the NeMo Agent Toolkit using Docker Compose, which will automatically pull the repository from GitHub and set up all services.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA API key (obtain from [build.nvidia.com](https://build.nvidia.com))

## Quick Start

1. **Create environment file:**
   Create a `.env` file in the same directory as `docker-compose.yml`:

   ```bash
   # Required API Keys
   NVIDIA_API_KEY=your_nvidia_api_key_here

   # Optional API Keys (uncomment and fill in if needed)
   # GITLAB_API_TOKEN=your_gitlab_token
   # PAGERDUTY_API_TOKEN=your_pagerduty_token
   # PAGERDUTY_API_URL=https://api.pagerduty.com
   # CONFLUENCE_API_TOKEN=your_confluence_token
   # CONFLUENCE_BASE_URL=https://confluence.nvidia.com/
   # LOKI_API_TOKEN=your_loki_token
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Check service status:**
   ```bash
   docker-compose ps
   ```

4. **View logs:**
   ```bash
   # View all logs
   docker-compose logs

   # View specific service logs
   docker-compose logs nemo-backend
   docker-compose logs nemo-ui
   ```

## Service Access Points

Once the services are running, you can access:

- **Frontend UI**: http://localhost:3000
- **ON_CALL Backend**: http://localhost:8000
- **FRIDAY Backend**: http://localhost:8001
- **SLACK Backend**: http://localhost:8002

## API Documentation

Each backend service provides Swagger documentation:

- ON_CALL API docs: http://localhost:8000/docs
- FRIDAY API docs: http://localhost:8001/docs
- SLACK API docs: http://localhost:8002/docs

## Testing the Setup

Test a simple calculation with the ON_CALL backend:

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "What is 2 + 2?",
    "use_knowledge_base": true
  }'
```

## Configuration

### Environment Variables

The following environment variables can be configured in your `.env` file:

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | NVIDIA API key for LLM access |
| `GITLAB_API_TOKEN` | No | GitLab API token for GitLab integrations |
| `PAGERDUTY_API_TOKEN` | No | PagerDuty API token |
| `PAGERDUTY_API_URL` | No | PagerDuty API URL (default: https://api.pagerduty.com) |
| `CONFLUENCE_API_TOKEN` | No | Confluence API token |
| `CONFLUENCE_BASE_URL` | No | Confluence base URL |
| `LOKI_API_TOKEN` | No | Loki API token |

### Port Configuration

If you need to change the default ports, modify the `ports` section in `docker-compose.yml`:

```yaml
ports:
  - "8000:8000"  # Change first number to desired host port
  - "8001:8001"
  - "8002:8002"
```

## Management Commands

### Start services
```bash
docker-compose up -d
```

### Stop services
```bash
docker-compose down
```

### Restart services
```bash
docker-compose restart
```

### Update services (rebuild from latest repo)
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### View service status
```bash
docker-compose ps
```

### Monitor logs in real-time
```bash
docker-compose logs -f
```

## Troubleshooting

### Services not starting

1. **Check environment variables:**
   ```bash
   docker-compose config
   ```

2. **Verify API keys:**
   Ensure your NVIDIA_API_KEY is valid and properly set in the `.env` file.

3. **Check Docker resources:**
   Ensure Docker has sufficient memory allocated (recommend 8GB+).

### Port conflicts

If ports 3000, 8000, 8001, or 8002 are already in use:

1. Stop conflicting services or change ports in `docker-compose.yml`
2. Update the frontend environment variables if changing backend ports

### Build issues

1. **Clear Docker cache:**
   ```bash
   docker system prune -a
   ```

2. **Rebuild without cache:**
   ```bash
   docker-compose build --no-cache
   ```

### Network issues

1. **Reset Docker networks:**
   ```bash
   docker-compose down
   docker network prune
   docker-compose up -d
   ```

## Development Mode

For development with hot reload, you can override the UI service:

```bash
# Override for development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Create `docker-compose.dev.yml`:
```yaml
version: '3.8'
services:
  nemo-ui:
    environment:
      - NODE_ENV=development
    command: npm run dev
```

## Health Checks

The services include health checks that monitor:
- Backend: HTTP endpoint availability
- UI: Frontend server response

Check health status:
```bash
docker-compose ps
```

## Logs and Debugging

### Access container shells
```bash
# Backend container
docker-compose exec nemo-backend bash

# UI container  
docker-compose exec nemo-ui bash
```

### View detailed logs
```bash
# Backend logs
docker-compose exec nemo-backend tail -f /app/logs/oncall_backend.log

# UI logs
docker-compose logs nemo-ui
```

## Updating

To update to the latest version from the repository:

```bash
# Stop services
docker-compose down

# Remove old images
docker-compose down --rmi all

# Rebuild and start
docker-compose up -d --build
```

## Additional Notes

- The setup automatically clones the repository from the specified GitHub URL and branch
- Git LFS files are automatically downloaded during the build process
- All Python dependencies and Node.js packages are installed automatically
- Backend logs are persisted in a Docker volume for debugging
- The UI automatically connects to all backend services via the internal Docker network 