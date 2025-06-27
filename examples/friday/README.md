# Friday AI Assistant

Friday is a comprehensive AI assistant that combines Confluence documentation search capabilities with Loki log analysis tools. It provides users with the ability to search for documentation and analyze system logs from a single interface.

## Features

### 🔍 Documentation Search (Confluence)
- **Text Search**: Search Confluence content using natural language queries
- **Title Search**: Find pages by title matching
- **Page Retrieval**: Get detailed content of specific Confluence pages
- **Multi-Space Support**: Search across multiple Confluence spaces
- **Link Preservation**: Always includes direct links to source documentation

### 📊 Log Analysis (Loki)
- **Natural Language Queries**: Analyze logs using plain English descriptions
- **Error Detection**: Automatically identify and categorize errors and warnings
- **Pattern Recognition**: Detect common patterns like timeouts, memory issues, etc.
- **Service Analysis**: Track log activity across different services
- **Time-based Filtering**: Search logs within specific time ranges
- **Advanced Filtering**: Filter by cluster, hostname, namespace, log level, etc.

### 🤝 Combined Intelligence
- **Cross-Reference**: Correlate log issues with documentation solutions
- **Contextual Help**: Find relevant docs when troubleshooting system issues
- **Comprehensive Reporting**: Combine log analysis with documented procedures

## Installation

1. Navigate to the Friday agent directory:
```bash
cd examples/friday
```

2. Install the package:
```bash
pip install -e .
```

## Configuration

### Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# Confluence Configuration
CONFLUENCE_BASE_URL=https://your-confluence.atlassian.net
CONFLUENCE_API_TOKEN=your_bearer_token_here

# Loki Configuration  
LOKI_API_TOKEN=your_loki_api_token_here
```

### Configuration File

The main configuration is in `configs/friday_config.yml`. Key settings include:

- **Confluence Settings**: Base URL, API token, search limits, space restrictions
- **Loki Settings**: Loki URL, time ranges, log limits, authentication
- **LLM Settings**: Model selection, temperature, token limits
- **Agent Behavior**: Retry settings, iteration limits, verbosity

## Usage Examples

### Documentation Search
```bash
# Search for deployment guides
aiq run configs/friday_config.yml "How do I deploy the application to production?"

# Find specific documentation
aiq run configs/friday_config.yml "Show me the API authentication documentation"
```

### Log Analysis
```bash
# Analyze recent errors
aiq run configs/friday_config.yml "Show me any errors in the last hour"

# Check specific service logs
aiq run configs/friday_config.yml "Analyze memory issues in the user-service"

# Investigate timeouts
aiq run configs/friday_config.yml "Find timeout errors in cluster gcp-cbf-cs-002"
```

### Combined Analysis
```bash
# Troubleshoot with both logs and docs
aiq run configs/friday_config.yml "I'm seeing database connection errors, help me troubleshoot"

# Find documentation for current issues
aiq run configs/friday_config.yml "Check logs for SSL errors and find related documentation"
```

## Tool Functions

### Confluence Tools
- `search_confluence(query)`: Search Confluence content
- `search_confluence_by_title(title_query)`: Search by page title
- `get_confluence_page(page_id)`: Get specific page content

### Loki Tools
- `analyze_logs(query, time_range, filters...)`: Comprehensive log analysis with multiple filter options

### Utility Tools
- `current_time()`: Get current timestamp for analysis context

## Advanced Configuration

### Confluence Spaces
To limit searches to specific Confluence spaces, uncomment and modify in the config:
```yaml
confluence_search:
  space_keys: ["TEAM", "DOCS", "PROJECT"]
```

### Loki Data Sources
Choose the appropriate Loki datasource UID in the configuration:
- loki-us-east-1: `ee0goqc2a64u8b`
- loki-us-west-2: `ce1n6e6v256v4c`  
- loki-eu-north-1: `fe1n6e6m54o3ka`
- loki-ap-northeast-1: `ee1n6e6vtm2o0c`

### Authentication Options

**Confluence**: Bearer token authentication (recommended)
**Loki**: Multiple options supported:
- API Token (recommended)
- Basic Authentication (username/password)
- Bearer Token

## Example Responses

### Documentation Search Response
```
**Answer:**
The deployment process involves three main steps: building the Docker image, 
pushing to the registry, and deploying via Kubernetes...

**Sources:**
- [Production Deployment Guide](https://confluence.example.com/pages/123) - Complete deployment procedures
- [Docker Build Process](https://confluence.example.com/pages/456) - Container building steps
```

### Log Analysis Response
```
📋 **Loki Log Analysis Results**

**Query:** errors in user-service
**Time Range:** 1h
**Total Entries:** 45

🚨 **Errors Found (3 total):**
**1.** [14:32:15] [user-service]
   Database connection timeout after 30 seconds

⚠️ **Warnings Found (12 total):**
**1.** [14:30:22] [user-service]
   High memory usage detected: 85% of allocated memory

🎯 **Recommendations:**
- Investigate the 3 error entries for root causes
- Check timeout configurations due to database connection issues
```

## Troubleshooting

### Common Issues

1. **Confluence Connection Issues**
   - Verify `CONFLUENCE_BASE_URL` is correct
   - Check that `CONFLUENCE_API_TOKEN` has proper permissions
   - Ensure network connectivity to Confluence instance

2. **Loki Query Failures**
   - Verify `LOKI_API_TOKEN` is valid
   - Check if the Loki URL and datasource UID are correct
   - Ensure proper authentication headers are configured

3. **No Results Found**
   - Try broader search terms
   - Check time ranges for log queries
   - Verify space permissions for Confluence searches

### Debug Mode
Enable verbose logging by setting `verbose: true` in the workflow configuration.

## Dependencies

- **httpx**: HTTP client for API requests
- **pydantic**: Data validation and settings management
- **python-dotenv**: Environment variable loading
- **python-dateutil**: Date parsing for time ranges

## License

Apache-2.0 License

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the configuration documentation
3. Examine log outputs for specific error messages
4. Ensure all environment variables are properly set 