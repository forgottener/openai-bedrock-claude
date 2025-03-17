# AWS Claude API 代理服务

## 项目简介

AWS Claude API 代理服务是一个基于Flask的Web应用，提供与OpenAI API兼容的接口，但实际使用AWS Bedrock上的Claude模型进行处理。这个代理服务使您能够使用现有的OpenAI API客户端无缝接入AWS上的Claude模型。

## 主要功能

- 支持多种Claude模型，包括Claude 3.7 Sonnet、Claude 3 Opus、Claude 3 Sonnet和Claude 3 Haiku等
- 提供与OpenAI API兼容的接口（/v1/completions和/v1/chat/completions）
- 支持Claude的思考模式（Thinking Mode）
- 自动参数验证和调整以符合AWS Bedrock要求
- 完善的重试机制，处理API限流和错误
- 强大的日志系统，支持日志轮转
- 流式输出支持

## 安装要求

### 系统依赖

- Python 3.8 或更高版本
- AWS账户并配置了Bedrock服务访问权限

### Python依赖

```bash
pip install flask boto3 tiktoken
```

## 配置说明

### 环境变量

- `DEBUG_MODE`: 设置为"true"启用详细日志记录（默认为"false"）

### AWS凭证

确保您的系统已配置AWS凭证，可以通过以下方式之一：

1. 使用AWS CLI: `aws configure`
2. 设置环境变量: `AWS_ACCESS_KEY_ID`和`AWS_SECRET_ACCESS_KEY`
3. 使用IAM角色（对EC2实例）

## 服务管理

### 使用systemd启动服务

服务文件路径: `/etc/systemd/system/claude-api.service`

```
[Unit]
Description=AWS Claude API Proxy Service
After=network.target

[Service]
User=your user
WorkingDirectory=/path/to #按需修改
ExecStart=/path/to/python3.8 /path/to/aws-claude.py #按需修改
Restart=always
RestartSec=10
Environment="DEBUG_MODE=false"

[Install]
WantedBy=multi-user.target
```

### 常用管理命令

```bash
# 启用服务（开机自启）
sudo systemctl enable claude-api.service

# 启动服务
sudo systemctl start claude-api.service

# 查看服务状态
sudo systemctl status claude-api.service

# 重启服务
sudo systemctl restart claude-api.service

# 停止服务
sudo systemctl stop claude-api.service

# 查看服务日志
sudo journalctl -u claude-api.service -f
```

## 日志管理

服务使用Python的RotatingFileHandler，日志会自动轮转：

- 日志文件位置: `/path/to/aws-claude.py/../logs/claude_service.log`
- 单文件大小上限: 20MB
- 最多保留5个文件（总计最多100MB）

### 查看日志

```bash
# 查看日志文件
cat /path/to/aws-claude.py/../logs/claude_service.log

# 实时跟踪日志
tail -f /path/to/aws-claude.py/../logs/claude_service.log

# 查看系统日志（systemd管理的服务日志）
sudo journalctl -u claude-api.service -f
```

## API使用示例

### 聊天完成示例（/v1/chat/completions）

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

### 文本完成示例（/v1/completions）

```bash
curl -X POST http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-7-sonnet",
    "prompt": "Write a short poem about clouds",
    "max_tokens": 200
  }'
```

### 思考模式示例

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

## 故障排除

### 日志问题

如果没有看到日志文件，请检查：

1. 权限问题：确保运行服务的用户有权限写入日志目录
   ```bash
   sudo chmod -R 755 /path/to/aws-claude.py/../logs/
   ```

2. 路径问题：确保日志目录存在
   ```bash
   sudo mkdir -p /path/to/aws-claude.py/../logs/
   ```

3. 检查systemd日志
   ```bash
   sudo journalctl -u claude-api.service -n 50
   ```

### API连接问题

如果服务运行但无法访问API：

1. 检查防火墙设置
   ```bash
   sudo iptables -L
   ```

2. 确认服务器监听端口
   ```bash
   sudo netstat -tulpn | grep python
   ```

3. 检查AWS凭证是否有效
   ```bash
   aws sts get-caller-identity
   ```

### 服务无法启动

如果服务无法启动，检查依赖是否安装：

```bash
pip install flask boto3 tiktoken
```

确保Python路径正确：

```bash
which python3.8
```

## 注意事项

- 服务默认在端口5000上运行
- 在生产环境中，建议使用反向代理（如Nginx）并配置HTTPS
- 定期监控日志文件大小和系统资源使用情况 