# AWS Claude API Proxy Service

## Project Overview

The AWS Claude API Proxy Service is a Flask-based web application that provides an OpenAI API-compatible interface while actually using Claude models on AWS Bedrock for processing. This proxy service allows you to seamlessly integrate existing OpenAI API clients with Claude models on AWS.

## Key Features

- Support for multiple Claude models, including Claude 3.7 Sonnet, Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku
- OpenAI API-compatible interfaces (/v1/completions and /v1/chat/completions)
- Support for Claude's Thinking Mode
- Automatic parameter validation and adjustment to meet AWS Bedrock requirements
- Robust retry mechanism for handling API throttling and errors
- Powerful logging system with log rotation
- Streaming output support

## Installation Requirements

### System Dependencies

- Python 3.8 or higher
- AWS account with Bedrock service access configured

### Python Dependencies

```bash
pip install flask boto3 tiktoken
```

## Configuration

### Environment Variables

- `DEBUG_MODE`: Set to "true" to enable detailed logging (default is "false")

### AWS Credentials

Ensure your system has AWS credentials configured using one of the following methods:

1. AWS CLI: `aws configure`
2. Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
3. IAM role (for EC2 instances)

## Service Management

### Starting the Service with systemd

Service file path: `/etc/systemd/system/claude-api.service`

```
[Unit]
Description=AWS Claude API Proxy Service
After=network.target

[Service]
User=your user name # edit me
WorkingDirectory=/path/to #edit me
ExecStart=/path/to/python3.8 /path/to/aws-claude.py #edit me
Restart=always
RestartSec=10
Environment="DEBUG_MODE=false"

[Install]
WantedBy=multi-user.target
```

### Common Management Commands

```bash
# Enable service (auto-start at boot)
sudo systemctl enable claude-api.service

# Start service
sudo systemctl start claude-api.service

# Check service status
sudo systemctl status claude-api.service

# Restart service
sudo systemctl restart claude-api.service

# Stop service
sudo systemctl stop claude-api.service

# View service logs
sudo journalctl -u claude-api.service -f
```

## Log Management

The service uses Python's RotatingFileHandler for automatic log rotation:

- Log file location: `/path/to/aws-claude.py/../logs/claude_service.log`
- Single file size limit: 20MB
- Maximum of 5 files retained (total maximum of 100MB)

### Viewing Logs

```bash
# View log file
cat /path/to/aws-claude.py/../logs/claude_service.log

# Real-time log tracking
tail -f /path/to/aws-claude.py/../logs/claude_service.log

# View system logs (for systemd-managed service)
sudo journalctl -u claude-api.service -f
```

## API Usage Examples

### Chat Completions Example (/v1/chat/completions)

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-7-sonnet",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you today?"
      }
    ],
    "max_tokens": 1000
  }'
```

### Text Completions Example (/v1/completions)

```bash
curl -X POST http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-7-sonnet",
    "prompt": "Write a short poem about clouds",
    "max_tokens": 200
  }'
```

### Thinking Mode Example

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-7-sonnet-thinking",
    "messages": [
      {
        "role": "user",
        "content": "What is the square root of 997 to 5 decimal places?"
      }
    ],
    "max_tokens": 2000
  }'
```

## Troubleshooting

### Log Issues

If you don't see any log files, check:

1. Permission issues: Ensure the user running the service (your user name) has permission to write to the log directory
   ```bash
   sudo chmod -R 755 /path/to/aws-claude.py/../logs/
   ```

2. Path issues: Ensure the log directory exists
   ```bash
   sudo mkdir -p /path/to/aws-claude.py/../logs/
   ```

3. Check systemd logs
   ```bash
   sudo journalctl -u claude-api.service -n 50
   ```

### API Connection Issues

If the service is running but the API is not accessible:

1. Check firewall settings
   ```bash
   sudo iptables -L
   ```

2. Confirm server listening port
   ```bash
   sudo netstat -tulpn | grep python
   ```

3. Verify AWS credentials are valid
   ```bash
   aws sts get-caller-identity
   ```

### Service Won't Start

If the service fails to start, check if dependencies are installed:

```bash
pip install flask boto3 tiktoken
```

Ensure Python path is correct:

```bash
which python3.8
```

## Important Notes

- Service runs on port 5000 by default
- For production environments, use a reverse proxy (like Nginx) and configure HTTPS
- Regularly monitor log file size and system resource usage

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 