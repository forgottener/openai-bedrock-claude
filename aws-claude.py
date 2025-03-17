"""
Copyright 2024 AWS Claude API Proxy Service Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from flask import Flask, request, Response, jsonify
import boto3
import json
import tiktoken
import time
import io
import random
import logging
import os
from botocore.exceptions import ClientError
from logging.handlers import RotatingFileHandler

# 调试模式配置
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'

# 使用绝对路径
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# 控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# 文件日志处理器（带轮转功能）
# 20MB一个文件，最多5个文件，总计最大100MB
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "claude_service.log"), 
    maxBytes=20*1024*1024,  # 20MB
    backupCount=4,          # 保留4个备份文件（加上当前文件共5个，最多100MB）
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# 添加处理器到logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 如果是调试模式，还可以配置更详细的boto3/botocore日志
if DEBUG_MODE:
    # 设置AWS SDK的日志级别
    boto3_logger = logging.getLogger('boto3')
    boto3_logger.setLevel(logging.DEBUG)
    boto3_logger.addHandler(file_handler)
    
    botocore_logger = logging.getLogger('botocore')
    botocore_logger.setLevel(logging.DEBUG)
    botocore_logger.addHandler(file_handler)
    
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.INFO)
    urllib3_logger.addHandler(file_handler)
    
    logger.debug("调试模式已启用 - 将输出详细日志信息")
else:
    logger.info("运行在正常模式 - 仅输出信息和警告日志")

app = Flask(__name__)
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# 全局配置
DEFAULT_MAX_TOKENS = 4096  # 默认输出token限制
MAX_OUTPUT_TOKENS = 128000  # Claude 3.7支持的最大输出tokens

# 模型映射字典
MODEL_MAPPING = {
    'claude-3-7-sonnet': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'claude-3-7-sonnet-thinking': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',  # 思考模式
    'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
    'claude-3-5-sonnet': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
    'claude-2': 'anthropic.claude-v2:1',
    'claude-instant': 'anthropic.claude-instant-v1'
}

# 默认模型 - Claude 3.7
DEFAULT_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'

# 重试配置
MAX_RETRIES = 5  # 最大重试次数
BASE_DELAY = 1  # 基础延迟时间（秒）
MAX_DELAY = 30  # 最大延迟时间（秒）

def log_request_info(endpoint, request_data):
    """记录请求信息，在调试模式下输出详细信息"""
    if DEBUG_MODE:
        # 创建一个可以安全打印的请求数据副本
        safe_data = request_data.copy() if request_data else {}
        
        # 如果有prompt或messages，可能会很长，截断它们
        if 'prompt' in safe_data and isinstance(safe_data['prompt'], str) and len(safe_data['prompt']) > 200:
            safe_data['prompt'] = safe_data['prompt'][:200] + '... [截断，完整内容仅在trace级别日志中]'
            
        if 'messages' in safe_data and isinstance(safe_data['messages'], list):
            for i, msg in enumerate(safe_data['messages']):
                if isinstance(msg, dict) and 'content' in msg and isinstance(msg['content'], str) and len(msg['content']) > 200:
                    safe_data['messages'][i]['content'] = msg['content'][:200] + '... [截断]'
        
        logger.debug(f"API请求 {endpoint}: {json.dumps(safe_data, ensure_ascii=False, indent=2)}")
        
        # 如果需要完整日志，可以设置更详细的级别
        logger.log(5, f"完整API请求 {endpoint}: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
    else:
        # 非调试模式只记录请求类型
        logger.info(f"处理API请求: {endpoint}")

def log_response_info(endpoint, response_data, is_error=False):
    """记录响应信息，在调试模式下输出详细信息"""
    if DEBUG_MODE:
        if is_error:
            logger.error(f"API错误响应 {endpoint}: {json.dumps(response_data, ensure_ascii=False)}")
        else:
            # 创建一个可以安全打印的响应数据副本
            safe_data = response_data.copy() if isinstance(response_data, dict) else {'data': str(response_data)[:200]}
            
            # 如果有choices，截断内容
            if 'choices' in safe_data and isinstance(safe_data['choices'], list):
                for i, choice in enumerate(safe_data['choices']):
                    if isinstance(choice, dict):
                        # 处理文本完成
                        if 'text' in choice and isinstance(choice['text'], str) and len(choice['text']) > 200:
                            safe_data['choices'][i]['text'] = choice['text'][:200] + '... [截断]'
                        
                        # 处理聊天完成
                        if 'message' in choice and isinstance(choice['message'], dict) and 'content' in choice['message']:
                            if isinstance(choice['message']['content'], str) and len(choice['message']['content']) > 200:
                                safe_data['choices'][i]['message']['content'] = choice['message']['content'][:200] + '... [截断]'
                        
                        # 处理思考内容
                        if 'thinking' in choice and isinstance(choice['thinking'], str) and len(choice['thinking']) > 200:
                            safe_data['choices'][i]['thinking'] = choice['thinking'][:200] + '... [截断]'
            
            logger.debug(f"API响应 {endpoint}: {json.dumps(safe_data, ensure_ascii=False)}")
    else:
        # 非调试模式只记录响应状态
        if is_error:
            logger.error(f"API错误响应: {endpoint}")
        else:
            logger.info(f"API请求完成: {endpoint}")

def count_tokens(text):
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def stream_iter_lines(stream):
    buffer = b''
    for chunk in stream:
        buffer += chunk
        while True:
            newline_pos = buffer.find(b'\n')
            if newline_pos == -1:
                break
            line, buffer = buffer[:newline_pos], buffer[newline_pos+1:]
            yield line
    if buffer:
        yield buffer

def validate_bedrock_request(params):
    """验证发送到Bedrock API的请求参数"""
    if DEBUG_MODE:
        logger.debug("正在验证Bedrock请求参数...")
        
    # 深度复制参数防止修改原始对象
    validated_params = params.copy() if params else {}
    
    # 检查是否使用思考模式
    is_thinking_enabled = False
    
    # 从参数中检查模型名称
    model_is_thinking = False
    
    # 直接从_model_name参数检查（由invoke_with_retry添加）
    if '_model_name' in validated_params and validated_params['_model_name'] == 'claude-3-7-sonnet-thinking':
        model_is_thinking = True
        logger.debug("检测到使用思考模式模型: claude-3-7-sonnet-thinking")
    # 从API请求中检查
    elif 'messages' in validated_params and isinstance(validated_params.get('messages'), list):
        # 这可能是API请求，提取model_name
        if 'model' in validated_params and validated_params['model'] == 'claude-3-7-sonnet-thinking':
            model_is_thinking = True
            logger.debug("检测到使用思考模式模型: claude-3-7-sonnet-thinking")
    
    # 检查思考模式参数
    if 'thinking' in validated_params:
        is_thinking_enabled = True
        thinking = validated_params['thinking']
        
        # 检查thinking格式是否正确
        if not isinstance(thinking, dict):
            logger.warning(f"思考参数类型错误: {type(thinking).__name__}，将重置为标准格式")
            thinking = {"type": "enabled", "budget_tokens": 4000}
            validated_params['thinking'] = thinking
        
        # 确保有type和budget_tokens字段
        if 'type' not in thinking or thinking.get('type') != 'enabled':
            logger.warning("思考参数缺少正确的type字段，将设置为'enabled'")
            thinking['type'] = 'enabled'
        
        # 验证budget_tokens
        if 'budget_tokens' not in thinking:
            logger.warning("思考参数缺少budget_tokens字段，将设置为默认值4000")
            thinking['budget_tokens'] = 4000
        else:
            try:
                # 确保budget_tokens是整数
                budget = int(thinking['budget_tokens'])
                if budget < 1024:
                    logger.warning(f"思考预算 {budget} 小于最低要求的1024，将调整为1024")
                    budget = 1024
                thinking['budget_tokens'] = budget
            except (ValueError, TypeError):
                logger.warning(f"思考预算值无效: {thinking.get('budget_tokens')}，将设置为默认值4000")
                thinking['budget_tokens'] = 4000
    
    # 如果启用了思考模式或使用思考模式模型，确保max_tokens足够大
    if (is_thinking_enabled or model_is_thinking) and 'max_tokens' in validated_params:
        try:
            max_tokens = int(validated_params['max_tokens'])
            if max_tokens < 1024:
                logger.warning(f"思考模式下max_tokens必须至少为1024，已从{max_tokens}调整为1024")
                validated_params['max_tokens'] = 1024
        except (ValueError, TypeError):
            logger.warning(f"max_tokens值无效: {validated_params.get('max_tokens')}，将设置为默认值4096")
            validated_params['max_tokens'] = 4096
    
    if model_is_thinking and 'thinking' not in validated_params:
        # 只有在用户没有明确禁用thinking的情况下添加
        if validated_params.get('thinking') is not False:
            logger.info("检测到思考模式模型但未指定thinking参数，自动添加")
            
            # 确保max_tokens足够大
            max_tokens = validated_params.get('max_tokens', DEFAULT_MAX_TOKENS)
            # 转为整数
            try:
                max_tokens = int(max_tokens)
            except (ValueError, TypeError):
                max_tokens = DEFAULT_MAX_TOKENS
                
            # 思考模式需要max_tokens至少为1024
            if max_tokens < 1024:
                logger.warning(f"思考模式模型要求max_tokens至少为1024，已将值从{max_tokens}调整为1024")
                max_tokens = 1024
                validated_params['max_tokens'] = max_tokens
            
            # 添加思考参数，预算设为max_tokens的80%但不小于1024
            thinking_budget = max(1024, min(int(max_tokens * 0.8), max_tokens - 1))
            validated_params['thinking'] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
    
    # 如果启用了思考模式，再次检查max_tokens是否足够
    if validated_params.get('thinking') and isinstance(validated_params['thinking'], dict):
        # 获取thinking预算
        thinking_budget = validated_params['thinking'].get('budget_tokens', 0)
        try:
            thinking_budget = int(thinking_budget)
        except (ValueError, TypeError):
            thinking_budget = 4000
            validated_params['thinking']['budget_tokens'] = thinking_budget
            
        # 获取max_tokens
        max_tokens = validated_params.get('max_tokens', DEFAULT_MAX_TOKENS)
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            max_tokens = DEFAULT_MAX_TOKENS
            
        # 确保max_tokens大于thinking预算
        if max_tokens <= thinking_budget:
            logger.warning(f"max_tokens({max_tokens})必须大于thinking预算({thinking_budget})，调整max_tokens为{thinking_budget + 1}")
            validated_params['max_tokens'] = thinking_budget + 1
        
        # 当启用思考模式时，必须移除top_p参数（AWS API限制）
        if 'top_p' in validated_params:
            logger.warning("思考模式下不能设置top_p参数，已自动移除")
            del validated_params['top_p']
    
    if DEBUG_MODE:
        logger.debug("请求参数验证完成")
        
    return validated_params

def invoke_with_retry(func, **kwargs):
    """使用指数退避重试机制调用 AWS Bedrock API"""
    retries = 0
    
    # 调试日志：记录API调用详情
    if DEBUG_MODE:
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        model_id = kwargs.get('modelId', 'unknown_model')
        logger.debug(f"准备调用 AWS Bedrock API: {func_name}, 模型: {model_id}")
        
        # 如果有请求体，记录格式化后的请求体，但需要确保敏感信息不会被完全记录
        if 'body' in kwargs and kwargs['body']:
            try:
                body_dict = json.loads(kwargs['body'])
                
                # 添加模型ID信息到请求参数中
                model_id = kwargs.get('modelId', '')
                # 从MODEL_MAPPING反向查找模型名称
                model_name = next((k for k, v in MODEL_MAPPING.items() if v == model_id), '')
                if model_name:
                    body_dict['_model_name'] = model_name
                
                # 验证请求参数
                body_dict = validate_bedrock_request(body_dict)
                
                # 从参数中移除临时添加的模型名称
                if '_model_name' in body_dict:
                    del body_dict['_model_name']
                
                # 更新请求体
                kwargs['body'] = json.dumps(body_dict)
                
                # 创建一个可以安全打印的请求体副本
                safe_body = body_dict.copy()
                
                # 如果有很长的文本，进行截断
                if 'messages' in safe_body and isinstance(safe_body['messages'], list):
                    for i, msg in enumerate(safe_body['messages']):
                        if 'content' in msg and isinstance(msg['content'], str) and len(msg['content']) > 200:
                            safe_body['messages'][i]['content'] = msg['content'][:200] + "... [截断]"
                
                logger.debug(f"请求参数: {json.dumps(safe_body, ensure_ascii=False, indent=2)}")
            except Exception as e:
                logger.debug(f"无法解析请求体进行日志记录: {str(e)}")
    
    while retries <= MAX_RETRIES:
        try:
            if DEBUG_MODE:
                start_time = time.time()
                
            response = func(**kwargs)
            
            if DEBUG_MODE:
                elapsed_time = time.time() - start_time
                logger.debug(f"API调用成功，耗时: {elapsed_time:.2f}秒")
                
                # 记录响应头信息，通常包含有用的调试信息如token计数
                if hasattr(response, 'get') and callable(response.get) and response.get('ResponseMetadata'):
                    headers = response['ResponseMetadata'].get('HTTPHeaders', {})
                    if headers:
                        logger.debug(f"响应头信息: {json.dumps(headers, ensure_ascii=False)}")
                        
                        # 特别记录token计数信息
                        token_info = {
                            'input_tokens': headers.get('x-amzn-bedrock-input-token-count'),
                            'output_tokens': headers.get('x-amzn-bedrock-output-token-count')
                        }
                        if any(token_info.values()):
                            logger.debug(f"Token用量: {json.dumps(token_info)}")
            
            return response
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_msg = e.response.get('Error', {}).get('Message', '')
            
            if DEBUG_MODE:
                logger.debug(f"AWS API 错误: {error_code} - {error_msg}")
                logger.debug(f"完整错误响应: {json.dumps(e.response, ensure_ascii=False)}")
            
            if error_code == 'ThrottlingException' and retries < MAX_RETRIES:
                # 计算等待时间（指数退避 + 随机抖动）
                delay = min(MAX_DELAY, BASE_DELAY * (2 ** retries) + random.uniform(0, 1))
                logger.warning(f"遇到节流限制 ({error_msg})，等待 {delay:.2f} 秒后进行第 {retries+1} 次重试...")
                time.sleep(delay)
                retries += 1
            else:
                logger.error(f"AWS API 错误: {error_code} - {error_msg}")
                raise
        except Exception as e:
            if DEBUG_MODE:
                logger.exception(f"调用 AWS API 时发生未知错误")
            else:
                logger.error(f"调用 AWS API 时发生未知错误: {str(e)}")
            raise

@app.route('/v1/completions', methods=['POST'])
def completions():
    """处理OpenAI兼容的文本补全API请求"""
    try:
        req = request.get_json()
        
        # 记录请求信息
        log_request_info("/v1/completions", req)
        
        prompt = req.get('prompt')
        
        # 确保prompt不为空
        if not prompt or not prompt.strip():
            error_response = {'error': 'Prompt cannot be empty'}
            log_response_info("/v1/completions", error_response, is_error=True)
            return jsonify(error_response), 400
            
        # 获取tokens配置
        max_tokens = req.get('max_tokens', DEFAULT_MAX_TOKENS)
        
        # 根据请求参数获取模型ID
        model_name = req.get('model', 'claude-3-7-sonnet')
        model_id = MODEL_MAPPING.get(model_name, DEFAULT_MODEL_ID)
        
        # 检查是否使用思考模式模型
        use_thinking_mode = model_name == 'claude-3-7-sonnet-thinking'
        
        # 如果用户选择了thinking模型但也明确设置了thinking=false，以用户设置为准
        if use_thinking_mode and req.get('thinking') is False:
            use_thinking_mode = False
            logger.info("用户选择了思考模式模型但明确禁用了思考功能，将以用户设置为准")
        
        # 如果用户明确设置了thinking=true，启用思考模式
        if req.get('thinking', False):
            use_thinking_mode = True
        
        # 支持超长输出模式（仅适用于Claude 3.7）
        enable_extended_output = req.get('enable_extended_output', False)
        
        # 检查tokens是否超过限制
        if max_tokens > MAX_OUTPUT_TOKENS:
            logger.warning(f"请求的max_tokens({max_tokens})超过了支持的最大值({MAX_OUTPUT_TOKENS})，将使用最大值")
            max_tokens = MAX_OUTPUT_TOKENS
            
        temperature = req.get('temperature', 1.0)
        top_p = req.get('top_p', 1.0)
        stop = req.get('stop', None)
        
        # 构建基本参数
        bed_params = {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        # 只有在非思考模式下才添加top_p参数
        if not use_thinking_mode:
            bed_params['top_p'] = top_p
        
        # 为Claude 3.7添加特殊功能支持
        if 'claude-3-7' in model_name:
            # 添加思考参数
            if use_thinking_mode:
                # 必须添加thinking参数，使用正确的对象格式
                thinking_budget = None
                if req.get('max_thinking_tokens', None):
                    thinking_budget = req.get('max_thinking_tokens')
                elif req.get('thinking_max_tokens', None):
                    thinking_budget = req.get('thinking_max_tokens')
                elif req.get('max_thinking_length', None):
                    thinking_budget = req.get('max_thinking_length')
                else:
                    # 如果未指定，默认使用max_tokens的一半作为思考预算，但不小于4000（推荐值）
                    thinking_budget = max(4000, int(max_tokens / 2))
                
                # 确保思考预算不小于最低要求的1024 tokens
                thinking_budget = max(1024, thinking_budget)
                
                # 确保思考预算小于max_tokens（AWS要求）
                if thinking_budget >= max_tokens:
                    # 如果思考预算大于等于最大输出限制，将其设为最大输出的80%
                    thinking_budget = int(max_tokens * 0.8)
                    logger.warning(f"思考预算不能大于等于最大输出限制，已调整为最大输出的80%: {thinking_budget} tokens")
                
                bed_params['thinking'] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
                logger.info(f"已启用Claude 3.7的思考模式，思考预算: {thinking_budget} tokens")
            
            # 支持超长输出模式（beta功能）
            if enable_extended_output and max_tokens > 64000:
                bed_params['anthropic_beta'] = ['output-128k-2025-02-19']
                logger.info(f"已启用Claude 3.7的超长输出模式 (128K tokens)")
            
            # 支持计算机使用功能（如果请求中指定）
            if req.get('enable_computer_use', False):
                if 'anthropic_beta' not in bed_params:
                    bed_params['anthropic_beta'] = []
                bed_params['anthropic_beta'].append('computer_20250212')
                logger.info(f"已启用Claude 3.7的计算机使用功能")
        
        if stop:
            bed_params['stop_sequences'] = stop

        if req.get('stream', False):
            body = json.dumps(bed_params)
            try:
                # 记录请求参数，便于调试
                if 'claude-3-7' in model_name and req.get('thinking', False):
                    logger.info(f"Claude 3.7思考模式请求参数: {json.dumps(bed_params, indent=2)}")
                
                # 使用重试机制调用流式API
                response = invoke_with_retry(
                    bedrock_runtime.invoke_model_with_response_stream,
                    modelId=model_id, 
                    body=body
                )
                def generate_stream():
                    last_completion = ""
                    for event in response['body']:
                        chunk = json.loads(event['chunk']['bytes'].decode('utf-8'))
                        delta = chunk.get('delta', {})
                        completion = delta.get('text', "")
                        stop_reason = chunk.get('stop_reason', None)
                        if completion:
                            finish_reason = stop_reason if stop_reason else None
                            event = {
                                "choices": [
                                    {
                                        "text": completion,
                                        "index": 0,
                                        "finish_reason": finish_reason
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(event)}\n\n"
                        if stop_reason:
                            break
                return Response(generate_stream(), mimetype='text/event-stream')
            except AttributeError as e:
                return jsonify({'error': f'Streaming not supported in this environment: {str(e)}'}), 500
        else:
            # 使用重试机制调用非流式API
            response = invoke_with_retry(
                bedrock_runtime.invoke_model,
                modelId=model_id, 
                body=json.dumps(bed_params)
            )
            bed_response = json.loads(response['body'].read().decode('utf-8'))
            
            # 从Bedrock响应中提取内容
            content = ""
            thinking_content = ""
            
            # 提取主要文本内容和思考内容
            if 'content' in bed_response and isinstance(bed_response['content'], list):
                for item in bed_response['content']:
                    if item.get('type') == 'text':
                        content += item.get('text', '')
            else:
                # 旧格式响应处理
                content = bed_response.get('completion', '')
                if not content:
                    content_obj = bed_response.get('content', None)
                    if isinstance(content_obj, list) and len(content_obj) > 0:
                        content = content_obj[0].get('text', '')
                    else:
                        content = str(content_obj) if content_obj else ''
            
            # 提取思考内容 (专门针对Claude 3.7)
            if 'claude-3-7' in model_name and use_thinking_mode:
                # 在消息对象中查找thinking块
                thinking_blocks = []
                
                # 首先检查顶级thinking字段
                if 'thinking' in bed_response:
                    thinking_content = bed_response.get('thinking', {}).get('text', '')
                
                # 如果没有找到，尝试在content中查找thinking类型的块
                if not thinking_content and 'content' in bed_response:
                    for item in bed_response.get('content', []):
                        if item.get('type') == 'thinking':
                            thinking_blocks.append(item.get('thinking', ''))
                    
                    if thinking_blocks:
                        thinking_content = ' '.join(thinking_blocks)
                
                if thinking_content:
                    logger.info(f"成功提取到思考内容，长度: {len(thinking_content)} 字符")
                else:
                    logger.warning(f"未能提取到思考内容，可能模型没有返回思考块或格式不符")
            
            stop_reason = bed_response.get('stop_reason', 'stop')

            # 计算token用量
            prompt_tokens = count_tokens(prompt)
            completion_tokens = count_tokens(content)
            thinking_tokens = count_tokens(thinking_content) if thinking_content else 0
            total_tokens = prompt_tokens + completion_tokens + thinking_tokens

            # 构建OpenAI格式的响应
            openai_response = {
                'id': 'some_id',
                'object': 'text_completion',
                'created': int(time.time()),
                'model': model_name,
                'choices': [{'text': content, 'index': 0, 'finish_reason': stop_reason}],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            }
            
            # 添加思考内容到响应中(如果有)
            if thinking_content:
                openai_response['choices'][0]['thinking'] = thinking_content
                openai_response['usage']['thinking_tokens'] = thinking_tokens

            # 记录响应信息
            log_response_info("/v1/completions", openai_response)
            return jsonify(openai_response)

    except Exception as e:
        error_msg = str(e)
        error_response = {'error': error_msg}
        
        # 记录错误信息
        if DEBUG_MODE:
            logger.exception("处理completions请求时发生错误")
        else:
            logger.error(f"处理completions请求时发生错误: {error_msg}")
            
        log_response_info("/v1/completions", error_response, is_error=True)
        return jsonify(error_response), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回可用的模型列表"""
    try:
        # 记录请求信息
        log_request_info("/v1/models", {"method": "GET"})
        
        models = []
        for model_name in MODEL_MAPPING.keys():
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic"
            })
        
        response = {
            "object": "list",
            "data": models
        }
        
        # 记录响应信息
        log_response_info("/v1/models", response)
        return jsonify(response)
    except Exception as e:
        error_msg = str(e)
        error_response = {'error': error_msg}
        
        # 记录错误信息
        if DEBUG_MODE:
            logger.exception("处理models请求时发生错误")
        else:
            logger.error(f"处理models请求时发生错误: {error_msg}")
            
        log_response_info("/v1/models", error_response, is_error=True)
        return jsonify(error_response), 500

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI兼容的聊天补全API"""
    try:
        req = request.get_json()
        
        # 记录请求信息
        log_request_info("/v1/chat/completions", req)
        
        messages = req.get('messages', [])
        
        # 根据请求参数获取模型ID
        model_name = req.get('model', 'claude-3-7-sonnet')
        model_id = MODEL_MAPPING.get(model_name, DEFAULT_MODEL_ID)
        
        # 检查是否使用思考模式模型
        use_thinking_mode = model_name == 'claude-3-7-sonnet-thinking'
        
        # 如果用户选择了thinking模型但也明确设置了thinking=false，以用户设置为准
        if use_thinking_mode and req.get('thinking') is False:
            use_thinking_mode = False
            logger.info("用户选择了思考模式模型但明确禁用了思考功能，将以用户设置为准")
        
        # 如果用户明确设置了thinking=true，启用思考模式
        if req.get('thinking', False):
            use_thinking_mode = True
        
        # 获取tokens配置
        max_tokens = req.get('max_tokens', DEFAULT_MAX_TOKENS)
        
        # 支持超长输出模式（仅适用于Claude 3.7）
        enable_extended_output = req.get('enable_extended_output', False)
        
        # 检查tokens是否超过限制
        if max_tokens > MAX_OUTPUT_TOKENS:
            logger.warning(f"请求的max_tokens({max_tokens})超过了支持的最大值({MAX_OUTPUT_TOKENS})，将使用最大值")
            max_tokens = MAX_OUTPUT_TOKENS

        # 过滤掉内容为空的消息，除了可能的最后一条助手消息
        filtered_messages = []
        for i, msg in enumerate(messages):
            # 获取消息内容，确保是字符串
            content = msg.get('content', '')
            if isinstance(content, list):
                # 如果内容是列表（多模态内容），检查是否有非空元素
                has_content = any(item.get('text', '') for item in content if item.get('type') == 'text')
            else:
                # 如果内容是字符串，直接检查是否为空
                has_content = bool(content and content.strip())
            
            # 如果消息有内容，或者是最后一条助手消息，则保留
            if has_content or (i == len(messages) - 1 and msg.get('role') == 'assistant'):
                filtered_messages.append(msg)
        
        # 确保至少有一条用户消息
        if not filtered_messages or not any(msg.get('role') == 'user' for msg in filtered_messages):
            return jsonify({'error': 'At least one user message with content is required'}), 400

        # 构建基本参数
        bed_params = {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': max_tokens,
            'temperature': req.get('temperature', 1.0),
            'messages': filtered_messages
        }
        
        # 只有在非思考模式下才添加top_p参数
        if not use_thinking_mode:
            bed_params['top_p'] = req.get('top_p', 1.0)
        
        # 为Claude 3.7添加扩展思考参数支持
        if 'claude-3-7' in model_name:
            # 添加思考参数
            if use_thinking_mode:
                # 必须添加thinking参数，使用正确的对象格式
                thinking_budget = None
                if req.get('max_thinking_tokens', None):
                    thinking_budget = req.get('max_thinking_tokens')
                elif req.get('thinking_max_tokens', None):
                    thinking_budget = req.get('thinking_max_tokens')
                elif req.get('max_thinking_length', None):
                    thinking_budget = req.get('max_thinking_length')
                else:
                    # 如果未指定，默认使用max_tokens的一半作为思考预算，但不小于4000（推荐值）
                    thinking_budget = max(4000, int(max_tokens / 2))
                
                # 确保思考预算不小于最低要求的1024 tokens
                thinking_budget = max(1024, thinking_budget)
                
                # 确保思考预算小于max_tokens（AWS要求）
                if thinking_budget >= max_tokens:
                    # 如果思考预算大于等于最大输出限制，将其设为最大输出的80%
                    thinking_budget = int(max_tokens * 0.8)
                    logger.warning(f"思考预算不能大于等于最大输出限制，已调整为最大输出的80%: {thinking_budget} tokens")
                
                bed_params['thinking'] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
                logger.info(f"已启用Claude 3.7的思考模式，思考预算: {thinking_budget} tokens")
            
            # 支持超长输出模式（beta功能）
            if enable_extended_output and max_tokens > 64000:
                bed_params['anthropic_beta'] = ['output-128k-2025-02-19']
                logger.info(f"已启用Claude 3.7的超长输出模式 (128K tokens)")
            
            # 支持计算机使用功能（如果请求中指定）
            if req.get('enable_computer_use', False):
                if 'anthropic_beta' not in bed_params:
                    bed_params['anthropic_beta'] = []
                bed_params['anthropic_beta'].append('computer_20250212')
                logger.info(f"已启用Claude 3.7的计算机使用功能")

        if req.get('stop', None):
            bed_params['stop_sequences'] = req.get('stop') if isinstance(req.get('stop'), list) else [req.get('stop')]

        if req.get('stream', False):
            body = json.dumps(bed_params)
            try:
                # 记录请求参数，便于调试
                if 'claude-3-7' in model_name and req.get('thinking', False):
                    logger.info(f"Claude 3.7思考模式请求参数: {json.dumps(bed_params, indent=2)}")
                
                # 使用重试机制调用流式API
                response = invoke_with_retry(
                    bedrock_runtime.invoke_model_with_response_stream,
                    modelId=model_id, 
                    body=body
                )
                def generate_stream():
                    for event in response['body']:
                        chunk = json.loads(event['chunk']['bytes'].decode('utf-8'))
                        
                        # 检查事件类型，处理不同类型的流式事件
                        event_type = chunk.get('type', '')
                        
                        # 处理不同类型的事件
                        if event_type == 'content_block_delta':
                            # 处理内容块增量
                            delta = chunk.get('delta', {})
                            
                            # 检查是否是思考块
                            if delta.get('type') == 'thinking_delta':
                                # 处理思考内容
                                thinking_text = delta.get('thinking', '')
                                if thinking_text:
                                    thinking_event = {
                                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"thinking": thinking_text},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(thinking_event)}\n\n"
                            
                            # 检查是否是文本块
                            elif delta.get('type') == 'text_delta':
                                # 处理普通文本内容
                                text = delta.get('text', '')
                                if text:
                                    event_data = {
                                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": text},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(event_data)}\n\n"
                        
                        # 处理消息结束
                        elif event_type == 'message_delta':
                            stop_reason = chunk.get('delta', {}).get('stop_reason')
                            if stop_reason:
                                finish_reason = 'stop' if stop_reason == 'end_turn' else stop_reason
                                event_data = {
                                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": finish_reason
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                                yield f"data: [DONE]\n\n"
                        
                        # 兼容旧版流式响应格式
                        elif not event_type:
                            # 旧的响应格式处理
                            delta = chunk.get('delta', {})
                            text = delta.get('text', "")
                            stop_reason = chunk.get('stop_reason', None)
                            
                            if text:
                                finish_reason = stop_reason if stop_reason else None
                                event_data = {
                                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": finish_reason
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                            
                            if stop_reason:
                                yield f"data: [DONE]\n\n"
                return Response(generate_stream(), mimetype='text/event-stream')
            except AttributeError as e:
                return jsonify({'error': f'Streaming not supported in this environment: {str(e)}'}), 500
            except Exception as e:
                return jsonify({'error': f'Stream error: {str(e)}'}), 500
        else:
            response = invoke_with_retry(
                bedrock_runtime.invoke_model,
                modelId=model_id, 
                body=json.dumps(bed_params)
            )
            bed_response = json.loads(response['body'].read().decode('utf-8'))
            
            # 从Bedrock响应中提取内容
            content = ""
            thinking_content = ""
            
            # 提取主要文本内容和思考内容
            if 'content' in bed_response and isinstance(bed_response['content'], list):
                for item in bed_response['content']:
                    if item.get('type') == 'text':
                        content += item.get('text', '')
            else:
                # 旧格式响应处理
                content = bed_response.get('completion', '')
                if not content:
                    content_obj = bed_response.get('content', None)
                    if isinstance(content_obj, list) and len(content_obj) > 0:
                        content = content_obj[0].get('text', '')
                    else:
                        content = str(content_obj) if content_obj else ''
            
            # 提取思考内容 (专门针对Claude 3.7)
            if 'claude-3-7' in model_name and use_thinking_mode:
                # 在消息对象中查找thinking块
                thinking_blocks = []
                
                # 首先检查顶级thinking字段
                if 'thinking' in bed_response:
                    thinking_content = bed_response.get('thinking', {}).get('text', '')
                
                # 如果没有找到，尝试在content中查找thinking类型的块
                if not thinking_content and 'content' in bed_response:
                    for item in bed_response.get('content', []):
                        if item.get('type') == 'thinking':
                            thinking_blocks.append(item.get('thinking', ''))
                    
                    if thinking_blocks:
                        thinking_content = ' '.join(thinking_blocks)
                
                if thinking_content:
                    logger.info(f"成功提取到思考内容，长度: {len(thinking_content)} 字符")
                else:
                    logger.warning(f"未能提取到思考内容，可能模型没有返回思考块或格式不符")
            
            stop_reason = bed_response.get('stop_reason', 'stop')

            # 计算token用量
            prompt_text = " ".join([msg.get('content', '') for msg in messages])
            prompt_tokens = count_tokens(prompt_text)
            completion_tokens = count_tokens(content)
            thinking_tokens = count_tokens(thinking_content) if thinking_content else 0
            total_tokens = prompt_tokens + completion_tokens + thinking_tokens

            # 构建OpenAI格式的响应
            openai_response = {
                'id': f"chatcmpl-{int(time.time() * 1000)}",
                'object': "chat.completion",
                'created': int(time.time()),
                'model': model_name,
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': content
                        },
                        'finish_reason': 'stop' if stop_reason == 'stop' else stop_reason
                    }
                ],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            }
            
            # 添加思考内容到响应中(如果有)
            if thinking_content:
                openai_response['choices'][0]['thinking'] = thinking_content
                openai_response['usage']['thinking_tokens'] = thinking_tokens

            # 记录响应信息
            log_response_info("/v1/chat/completions", openai_response)
            return jsonify(openai_response)

    except Exception as e:
        error_msg = str(e)
        error_response = {'error': error_msg}
        
        # 记录错误信息
        if DEBUG_MODE:
            logger.exception("处理chat_completions请求时发生错误")
        else:
            logger.error(f"处理chat_completions请求时发生错误: {error_msg}")
            
        log_response_info("/v1/chat/completions", error_response, is_error=True)
        return jsonify(error_response), 500

def is_claude_37_configured():
    """检查Claude 3.7模型是否正确配置"""
    models_to_check = ['claude-3-7-sonnet', 'claude-3-7-sonnet-thinking']
    
    for model_name in models_to_check:
        if model_name in MODEL_MAPPING:
            model_id = MODEL_MAPPING.get(model_name, '')
            if not model_id.startswith('us.'):
                logger.warning(f"警告: {model_name}需要使用us.前缀或推理配置文件ARN，当前配置可能不正确")
                return False
    
    return True

if __name__ == '__main__':
    # 启动信息标头
    logger.info("=" * 50)
    logger.info("启动 Claude API 代理服务器...")
    logger.info("=" * 50)
    
    # 调试模式信息
    if DEBUG_MODE:
        logger.info("调试模式已启用 - 将输出详细的请求和响应日志")
        logger.debug("调试模式配置检查:")
        logger.debug(f"- 日志级别: {logging.getLevelName(logger.level)}")
        logger.debug(f"- 可用模型数量: {len(MODEL_MAPPING)}")
    
    # 基本服务信息
    logger.info(f"可用模型: {', '.join(MODEL_MAPPING.keys())}")
    logger.info(f"默认模型: {[k for k, v in MODEL_MAPPING.items() if v == DEFAULT_MODEL_ID][0]}")
    logger.info(f"默认输出Tokens: {DEFAULT_MAX_TOKENS}")
    logger.info(f"最大支持输出Tokens: {MAX_OUTPUT_TOKENS}")
    
    # 检查Claude 3.7配置
    if 'claude-3-7-sonnet' in MODEL_MAPPING:
        if is_claude_37_configured():
            logger.info("Claude 3.7 Sonnet配置正确")
        else:
            logger.warning("Claude 3.7 Sonnet配置可能不正确，请检查model格式")
    
    # 服务器配置信息
    logger.info(f"服务器地址: 0.0.0.0")
    logger.info(f"设置: 线程模式={True}, 进程数={1}")
    logger.info("=" * 50)
    
    # 启动服务器
    app.run(host='0.0.0.0', threaded=True, processes=1)