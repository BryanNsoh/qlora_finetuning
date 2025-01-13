import os
import json
import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    Generic,
)

from dotenv import load_dotenv
# Load .env (override any pre-existing environment variables with what's in .env)
load_dotenv(override=True)

import logfire
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

# Core Pydantic AI
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.vertexai import VertexAIModel
from pydantic_ai.exceptions import UserError

# Configure logfire (optional)
logfire.configure(send_to_logfire="if-token-present")

T = TypeVar("T", bound=BaseModel)

# ----------------------------------------------------------------------------
# Response Models
# ----------------------------------------------------------------------------

class BatchMetadata(BaseModel):
    """Metadata for batch processing jobs."""
    batch_id: str
    input_file_id: str
    status: str
    created_at: datetime
    last_updated: datetime
    num_requests: int
    error: Optional[str] = None
    output_file_path: Optional[str] = None


class BatchResult(BaseModel):
    """Results from batch processing."""
    metadata: BatchMetadata
    results: List[Dict[str, Union[str, BaseModel]]]


class SimpleResponse(BaseModel):
    """Simple response model for testing."""
    content: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class MathResponse(BaseModel):
    """Response model for math problems."""
    answer: Optional[float] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class PersonResponse(BaseModel):
    """Response model for person descriptions."""
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    occupation: Optional[str] = None
    skills: Optional[List[str]] = None


class UnifiedResponse(BaseModel, Generic[T]):
    """A unified response envelope."""
    success: bool
    data: Optional[Union[T, List[T], BatchResult]] = None
    error: Optional[str] = None
    original_prompt: Optional[str] = None

# ----------------------------------------------------------------------------
# Handler Class
# ----------------------------------------------------------------------------

class UnifiedLLMHandler:
    """
    A unified handler for processing single or multiple prompts with typed responses,
    optional batch mode, and multiple LLM providers.

    The user MUST pass provider: in the model string (e.g. 'openai:gpt-4'),
    or we raise an error. This eliminates ambiguous model name guessing.
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        batch_output_dir: str = "batch_output",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        :param requests_per_minute: If specified, uses an AsyncLimiter to throttle requests.
        :param batch_output_dir: Directory for saving batch output JSONL.
        :param openai_api_key: OpenAI API key override (falls back to OPENAI_API_KEY if None).
        :param openrouter_api_key: OpenRouter API key override (falls back to OPENROUTER_API_KEY).
        :param deepseek_api_key: DeepSeek API key override (falls back to DEEPSEEK_API_KEY).
        :param anthropic_api_key: Anthropic API key override (falls back to ANTHROPIC_API_KEY).
        :param gemini_api_key: Gemini (Generative Language API) key override (falls back to GEMINI_API_KEY).
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self.rate_limiter = (
            AsyncLimiter(requests_per_minute, 60) if requests_per_minute else None
        )
        self.batch_output_dir = Path(batch_output_dir)
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)

    def _build_model_instance(self, model_str: str) -> Model:
        """
        If model_str is not prefixed with a recognized provider, raise an error.
        Otherwise, build and return the correct pydantic_ai model instance.
        """
        # 1) Check prefix
        if ":" not in model_str:
            raise UserError(
                "Model string must start with a recognized prefix, "
                "e.g. 'openai:gpt-4', 'openrouter:anthropic/claude-3.5', "
                "'deepseek:deepseek-chat', 'anthropic:claude-2', 'gemini:gemini-1.5-flash', "
                "'vertexai:gemini-1.5-flash'."
            )

        provider, real_model_name = model_str.split(":", 1)
        provider = provider.strip().lower()
        real_model_name = real_model_name.strip()

        # 2) Based on prefix, create instance
        if provider == "openai":
            if not self.openai_api_key:
                raise UserError("No OpenAI API key set. Provide openai_api_key= or set OPENAI_API_KEY.")
            return OpenAIModel(real_model_name, api_key=self.openai_api_key)

        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise UserError("No OpenRouter API key set. Provide openrouter_api_key= or set OPENROUTER_API_KEY.")
            return OpenAIModel(
                real_model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )

        elif provider == "deepseek":
            if not self.deepseek_api_key:
                raise UserError("No DeepSeek API key set. Provide deepseek_api_key= or set DEEPSEEK_API_KEY.")
            return OpenAIModel(
                real_model_name,
                base_url="https://api.deepseek.com",
                api_key=self.deepseek_api_key,
            )

        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise UserError("No Anthropic API key set. Provide anthropic_api_key= or set ANTHROPIC_API_KEY.")
            return AnthropicModel(real_model_name, api_key=self.anthropic_api_key)

        elif provider == "gemini":
            if not self.gemini_api_key:
                raise UserError("No Gemini API key set. Provide gemini_api_key= or set GEMINI_API_KEY.")
            return GeminiModel(real_model_name, api_key=self.gemini_api_key)

        elif provider == "vertexai":
            # Typically uses GCP credentials automatically.
            return VertexAIModel(real_model_name)

        else:
            raise UserError(
                f"Unrecognized provider prefix: {provider}. "
                f"Must be one of: openai, openrouter, deepseek, anthropic, gemini, vertexai."
            )

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: str,
        response_type: Type[T],
        *,
        system_message: Union[str, Sequence[str]] = (),
        batch_size: int = 1000,
        batch_mode: bool = False,
        retries: int = 1,
    ) -> UnifiedResponse[Union[T, List[T], BatchResult]]:
        """
        Main entry point for processing user prompts with typed responses.

        :param prompts: The prompt or list of prompts.
        :param model: Must be "provider:model_name", e.g. "openai:gpt-4".
        :param response_type: A pydantic BaseModel for typed responses, e.g. SimpleResponse.
        :param system_message: Optional system message(s) to guide the model.
        :param batch_size: If multiple prompts are provided, process them in chunks.
        :param batch_mode: If True, uses the OpenAI batch API (only for openai: models).
        :param retries: Number of times to retry on certain exceptions.
        """
        with logfire.span("llm_processing"):
            original_prompt_for_error = None
            if isinstance(prompts, str):
                original_prompt_for_error = prompts
            elif isinstance(prompts, list) and prompts:
                original_prompt_for_error = prompts[0]

            try:
                if prompts is None:
                    raise UserError("Prompts cannot be None.")
                if isinstance(prompts, list) and len(prompts) == 0:
                    raise UserError("Prompts list cannot be empty.")

                model_instance = self._build_model_instance(model)

                agent = Agent(
                    model_instance,
                    result_type=response_type,
                    system_prompt=system_message,
                    retries=retries,
                )

                if batch_mode:
                    # Only openai: supports batch API
                    if not isinstance(model_instance, OpenAIModel):
                        raise UserError(
                            "Batch API mode is only supported for openai:... models."
                        )
                    batch_result = await self._process_batch(agent, prompts, response_type)
                    return UnifiedResponse(success=True, data=batch_result)

                if isinstance(prompts, str):
                    data = await self._process_single(agent, prompts)
                    return UnifiedResponse(success=True, data=data)
                else:
                    data = await self._process_multiple(agent, prompts, batch_size)
                    return UnifiedResponse(success=True, data=data)

            except UserError as e:
                full_trace = traceback.format_exc()
                error_msg = f"UserError: {e}\nFull Traceback:\n{full_trace}"
                with logfire.span("error_handling", error=str(e), error_type="user_error"):
                    return UnifiedResponse(
                        success=False,
                        error=error_msg,
                        original_prompt=original_prompt_for_error,
                    )
            except Exception as e:
                full_trace = traceback.format_exc()
                error_msg = f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                with logfire.span("error_handling", error=str(e), error_type="unexpected_error"):
                    return UnifiedResponse(
                        success=False,
                        error=error_msg,
                        original_prompt=original_prompt_for_error,
                    )

    async def _process_single(self, agent: Agent, prompt: str) -> T:
        """
        Process a single prompt with optional rate limiting.
        """
        with logfire.span("process_single"):
            if self.rate_limiter:
                async with self.rate_limiter:
                    result = await agent.run(prompt)
            else:
                result = await agent.run(prompt)
            return result.data

    async def _process_multiple(
        self, agent: Agent, prompts: List[str], batch_size: int
    ) -> List[T]:
        """
        Process multiple prompts in chunks (batch_size) using asyncio.gather for concurrency.
        """
        results = []
        with logfire.span("process_multiple"):
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]

                async def process_prompt(p: str) -> T:
                    if self.rate_limiter:
                        async with self.rate_limiter:
                            res = await agent.run(p)
                    else:
                        res = await agent.run(p)
                    return res.data

                batch_results = await asyncio.gather(*(process_prompt(p) for p in batch))
                results.extend(batch_results)
        return results

    async def _process_batch(
        self, agent: Agent, prompts: List[str], response_type: Type[T]
    ) -> BatchResult:
        """
        Specialized method for the OpenAI Batch API workflow.
        Writes JSONL, uploads to OpenAI, polls for completion,
        and returns a BatchResult with typed responses.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.batch_output_dir / f"batch_{timestamp}.jsonl"

        with logfire.span("process_batch"):
            with batch_file.open("w") as f:
                for i, prompt in enumerate(prompts):
                    request = {
                        "custom_id": f"req_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": agent.model.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    }
                    f.write(json.dumps(request) + "\n")

            batch_upload = await agent.model.client.files.create(
                file=batch_file.open("rb"), purpose="batch"
            )
            batch = await agent.model.client.batches.create(
                input_file_id=batch_upload.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            metadata = BatchMetadata(
                batch_id=batch.id,
                input_file_id=batch_upload.id,
                status="in_progress",
                created_at=datetime.now(),
                last_updated=datetime.now(),
                num_requests=len(prompts),
            )

            while True:
                status = await agent.model.client.batches.retrieve(batch.id)
                metadata.status = status.status
                metadata.last_updated = datetime.now()

                if status.status == "completed":
                    break
                elif status.status in ["failed", "canceled"]:
                    metadata.error = f"Batch failed with status: {status.status}"
                    return BatchResult(metadata=metadata, results=[])

                await asyncio.sleep(10)

            output_file = self.batch_output_dir / f"batch_{batch.id}_results.jsonl"
            result_content = await agent.model.client.files.content(
                status.output_file_id
            )
            with output_file.open("wb") as f:
                f.write(result_content.content)
            metadata.output_file_path = str(output_file)

            results = []
            with output_file.open() as f:
                for line, prompt in zip(f, prompts):
                    data = json.loads(line)
                    try:
                        content = data["response"]["body"]["choices"][0]["message"]["content"]
                        r = response_type.construct()
                        if "content" in response_type.model_fields:
                            setattr(r, "content", content)
                        if "confidence" in response_type.model_fields:
                            setattr(r, "confidence", 0.95)
                        results.append({"prompt": prompt, "response": r})
                    except Exception as e:
                        full_trace = traceback.format_exc()
                        error_msg = f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                        results.append({"prompt": prompt, "error": error_msg})

            return BatchResult(metadata=metadata, results=results)


# ----------------------------------------------------------------------------
# Testing / Demonstration
# ----------------------------------------------------------------------------

async def run_tests():
    """
    Demonstrates usage with single, multiple, and batch prompts,
    for different providers. The user must pass "provider:model".
    """
    handler = UnifiedLLMHandler(
        requests_per_minute=2000,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Test 1: Single prompt (openai)
    single_result = await handler.process(
        prompts="Explain quantum computing in simple terms",
        model="openai:gpt-3.5-turbo",
        response_type=SimpleResponse,
    )
    print("\nTest 1 (Single Prompt, openai:gpt-3.5-turbo):")
    print("Success?", single_result.success)
    if single_result.success and single_result.data:
        print("Data:", single_result.data.model_dump(mode="python"))
    else:
        print("Error:", single_result.error)

    # Test 2: Multiple prompts (gemini)
    math_questions = ["What is 127+345?", "Calculate 15% of 2500", "What is sqrt(169)?"]
    multi_result = await handler.process(
        prompts=math_questions,
        model="gemini:gemini-1.5-flash-8b",
        response_type=MathResponse,
        system_message="You are a precise mathematical assistant. Always show your reasoning.",
    )
    print("\nTest 2 (Multiple Prompts, gemini:gemini-1.5-flash-8b):")
    print("Success?", multi_result.success)
    if multi_result.success and multi_result.data:
        for d in multi_result.data:
            print(d.model_dump(mode="python"))
    else:
        print("Error:", multi_result.error)

    # Test 3: Parallel processing (openai)
    person_prompts = [
        "Describe a software engineer in Silicon Valley",
        "Describe a chef in a Michelin star restaurant",
    ]
    person_result = await handler.process(
        prompts=person_prompts,
        model="openai:gpt-4o-mini",
        response_type=PersonResponse,
        batch_size=2,
    )
    print("\nTest 3 (Parallel, openai:gpt-4o-mini):")
    print("Success?", person_result.success)
    if person_result.success and isinstance(person_result.data, list):
        for d in person_result.data:
            print(d.model_dump(mode="python"))
    else:
        print("Error:", person_result.error)

    # Test 4: Batch mode (openai)
    simple_prompts = [f"Write a short story about {i}" for i in range(3)]
    batch_result = await handler.process(
        prompts=simple_prompts,
        model="openai:gpt-3.5-turbo",
        response_type=SimpleResponse,
        batch_mode=True,
    )
    print("\nTest 4 (Batch API, openai:gpt-3.5-turbo):")
    print("Success?", batch_result.success)
    if batch_result.success and batch_result.data:
        print(batch_result.data.model_dump(mode="python"))
    else:
        print("Error:", batch_result.error)

    # Test 5: Attempting batch mode with a non-OpenAI model => error
    batch_result_wrong = await handler.process(
        prompts=simple_prompts,
        model="gemini:gemini-1.5-flash-8b",
        response_type=SimpleResponse,
        batch_mode=True,
    )
    print("\nTest 5 (Batch API with gemini => error):")
    print("Success?", batch_result_wrong.success)
    print("Error:", batch_result_wrong.error)

    # Test 6: Invalid prefix => error
    invalid_model_result = await handler.process(
        prompts="Test prompt", model="gpt-4o-mini", response_type=SimpleResponse
    )
    print("\nTest 6 (Invalid prefix => error):")
    print("Success?", invalid_model_result.success)
    print("Error:", invalid_model_result.error)

    # Test 7: OpenRouter usage
    openrouter_result = await handler.process(
        prompts="Summarize the differences between YAGNI and KISS principles.",
        model="openrouter:anthropic/claude-3.5-sonnet",
        response_type=SimpleResponse,
    )
    print("\nTest 7 (OpenRouter usage):")
    print("Success?", openrouter_result.success)
    print("Output:", openrouter_result.data if openrouter_result.success else openrouter_result.error)

    # Test 8: DeepSeek usage
    deepseek_result = await handler.process(
        prompts="Hello from DeepSeek! Tell me a fun fact about software engineering.",
        model="deepseek:deepseek-chat",
        response_type=SimpleResponse,
    )
    print("\nTest 8 (DeepSeek usage):")
    print("Success?", deepseek_result.success)
    print("Output:", deepseek_result.data if deepseek_result.success else deepseek_result.error)

    # Test 9: Anthropic usage
    anthro_result = await handler.process(
        prompts="Explain the Single Responsibility Principle in detail.",
        model="anthropic:claude-3-5-sonnet-20241022",
        response_type=SimpleResponse,
    )
    print("\nTest 9 (Anthropic usage):")
    print("Success?", anthro_result.success)
    print("Output:", anthro_result.data if anthro_result.success else anthro_result.error)


if __name__ == "__main__":
    asyncio.run(run_tests())
