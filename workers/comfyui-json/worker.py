import base64
import json
import os
import re

from aiohttp import web, ClientResponse
from typing import Union

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# vLLM model configuration for Qwen3-VL
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18000
MODEL_LOG_FILE             = '/var/log/portal/vllm.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# vLLM-specific log messages
MODEL_LOAD_LOG_MSG = [
    "Application startup complete.",
]

MODEL_ERROR_LOG_MSGS = [
    "INFO exited: vllm",
    "RuntimeError: Engine",
    "Traceback (most recent call last):"
]

MODEL_INFO_LOG_MSGS = [
    '"message":"Download'
]

# Image paths (relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATHS = [
    os.path.join(SCRIPT_DIR, "present.jpg"),
    os.path.join(SCRIPT_DIR, "not-present.jpg"),
]

# The condition to evaluate
CONDITION = "A baby is present"

# Validation thresholds
EXPECTED_SCORES = {
    "present.jpg": {"min": 0.7, "max": 1.0},      # Should return high confidence
    "not-present.jpg": {"min": 0.0, "max": 0.3},  # Should return low confidence
}

def load_image_as_base64(image_path: str) -> str:
    """Load an image file and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_prompt(condition: str) -> str:
    """Build the analysis prompt with the given condition."""
    return f"""Analyze the image and return a single number between 0.0 and 1.0.
The number represents how confident you are that the following statement is true.
Do not include any other text.

Statement:
{condition}"""

def request_parser(request):
    """Extract the input data from the request."""
    data = request
    if request.get("input") is not None:
        data = request.get("input")
    return data

def extract_score(text: str) -> float:
    """Extract a float score from model response text."""
    text = text.strip()
    # Try direct float parse first
    try:
        return float(text)
    except ValueError:
        pass
    # Try to find a float pattern in the text
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract score from: {text}")

async def validate_response(
    client_request: web.Request,
    model_response: ClientResponse
) -> Union[web.Response, web.StreamResponse]:
    """Validate that the model returns expected confidence scores."""
    # Read the response body
    response_body = await model_response.read()
    content_type = model_response.content_type or "application/json"

    # Read the request to get the image name
    request_body = await client_request.text()
    request_data = json.loads(request_body)
    image_name = request_data.get("_image_name", "unknown")

    try:
        response_data = json.loads(response_body)

        # Extract the model's response content
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Extract the confidence score
        score = extract_score(content)

        # Validate against expected range
        if image_name in EXPECTED_SCORES:
            expected = EXPECTED_SCORES[image_name]
            if not (expected["min"] <= score <= expected["max"]):
                error_msg = (
                    f"VALIDATION FAILED for {image_name}: "
                    f"score {score} not in expected range [{expected['min']}, {expected['max']}]"
                )
                print(f"[VALIDATION] {error_msg}")
                return web.Response(
                    body=json.dumps({"error": error_msg, "score": score, "image": image_name}),
                    status=400,
                    content_type="application/json"
                )
            else:
                print(f"[VALIDATION] PASSED for {image_name}: score {score} in range [{expected['min']}, {expected['max']}]")
        else:
            print(f"[VALIDATION] WARNING: No expected score defined for {image_name}, score was {score}")

        # Return the original response (validation passed)
        return web.Response(
            body=response_body,
            status=model_response.status,
            content_type=content_type
        )

    except Exception as e:
        error_msg = f"VALIDATION ERROR for {image_name}: {str(e)}"
        print(f"[VALIDATION] {error_msg}")
        return web.Response(
            body=json.dumps({"error": error_msg, "image": image_name}),
            status=400,
            content_type="application/json"
        )

# Build benchmark dataset with both images
benchmark_dataset = []
for image_path in IMAGE_PATHS:
    image_name = os.path.basename(image_path)
    image_base64 = load_image_as_base64(image_path)

    benchmark_data = {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct"),
        "_image_name": image_name,  # For validation tracking
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": build_prompt(CONDITION)
                    }
                ]
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    benchmark_dataset.append({"input": benchmark_data})

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            response_generator=validate_response,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
                concurrency=1,
                runs=1
            )
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
