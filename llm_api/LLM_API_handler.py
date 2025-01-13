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
    Literal,
    Generic,
)

import logfire
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

# Pydantic AI imports
from pydantic_ai import Agent
from pydantic_ai.models import Model, KnownModelName, infer_model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.exceptions import UserError

try:
    # Anthropic is optional, so only import if available
    from pydantic_ai.models.anthropic import AnthropicModel
except ImportError:
    # If anthropic isn't installed, we can handle that gracefully if needed.
    AnthropicModel = None

# Configure logfire for demonstration
logfire.configure(send_to_logfire="if-token-present")

T = TypeVar("T", bound=BaseModel)
ModelProvider = Literal["openai", "groq", "vertexai"]

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
    A unified handler for processing single or multiple prompts with typed responses
    and optional batch mode, supporting multiple LLM providers.

    Follows principles of clarity (KISS), DRY, single responsibility (each method
    clearly handles one concern: single prompt, multiple prompts, or batch).
    """

    # Known short-hands => provider. This is used if we have "gpt-4o" etc.
    MODEL_PREFIXES = {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "o1-preview": "openai",
        "o1-mini": "openai",
        "llama-3.1-70b-versatile": "groq",
        "llama3-groq-70b-8192-tool-use-preview": "groq",
        "llama-3.1-70b-specdec": "groq",
        "llama-3.1-8b-instant": "groq",
        "llama-3.2-1b-preview": "groq",
        "llama-3.2-3b-preview": "groq",
        "mixtral-8x7b-32768": "groq",
        "gemma2-9b-it": "groq",
        "gemma-7b-it": "groq",
        # Models that have no default prefix
        "gemini-1.5-flash": None,
        "gemini-1.5-pro": None,
        "gemini-1.0-pro": None,
        "gemini-1.5-flash-8b": None,
        "vertex-gemini-1.5-flash": "vertexai",
        "vertex-gemini-1.5-pro": "vertexai",
        "vertex-gemini-1.5-flash-8b": "vertexai",
    }

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
        :param requests_per_minute: If specified, an AsyncLimiter is used to throttle requests.
        :param batch_output_dir: Directory where batch output JSONL is saved.
        :param openai_api_key: Optional explicit key for direct OpenAI usage.
        :param openrouter_api_key: Optional explicit key for OpenRouter usage.
        :param deepseek_api_key: Optional explicit key for DeepSeek usage.
        :param anthropic_api_key: Optional explicit key for Anthropic usage.
        """
        # DRY principle: store environment or user-provided keys
        # If the user doesn't provide the key, fallback to env variable
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

    def _map_short_model_name(self, model_name: str) -> str:
        """
        If the user-provided model doesn't contain a colon and is recognized in MODEL_PREFIXES,
        add the prefix. For example, "gpt-4o-mini" => "openai:gpt-4o-mini".
        Otherwise return as-is.
        """
        if ":" in model_name:
            return model_name
        prefix = self.MODEL_PREFIXES.get(model_name)
        return model_name if prefix is None else f"{prefix}:{model_name}"

    def _build_model_instance(
        self, full_model_str: str
    ) -> Model:
        """
        Decide which provider to use based on `full_model_str` prefix.
        Return a pydantic_ai Model instance.

        Follows KISS for clarity: each prefix has a simple if/elif. 
        DRY: we unify repeated logic for environment keys.
        """
        # Anthropic usage, e.g. "anthropic:claude-2"
        if full_model_str.startswith("anthropic:"):
            if AnthropicModel is None:
                raise UserError(
                    "AnthropicModel not available. Please install pydantic-ai with the `[anthropic]` extra."
                )
            real_model = full_model_str[len("anthropic:"):]
            if not self.anthropic_api_key:
                raise UserError(
                    "No Anthropic API key found. Either set ANTHROPIC_API_KEY in .env or pass anthropic_api_key="
                )
            return AnthropicModel(real_model, api_key=self.anthropic_api_key)

        # OpenRouter usage, e.g. "openrouter:anthropic/claude-3.5-sonnet"
        if full_model_str.startswith("openrouter:"):
            real_model = full_model_str[len("openrouter:"):]
            if not self.openrouter_api_key:
                raise UserError(
                    "No OpenRouter API key found. Either set OPENROUTER_API_KEY in .env or pass openrouter_api_key="
                )
            return OpenAIModel(
                real_model,
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )

        # DeepSeek usage, e.g. "deepseek:deepseek-chat"
        if full_model_str.startswith("deepseek:"):
            real_model = full_model_str[len("deepseek:"):]
            if not self.deepseek_api_key:
                raise UserError(
                    "No DeepSeek API key found. Either set DEEPSEEK_API_KEY in .env or pass deepseek_api_key="
                )
            return OpenAIModel(
                real_model,
                base_url="https://api.deepseek.com",
                api_key=self.deepseek_api_key,
            )
            
        # Gemini usage, e.g. "gemini:gemini-1.5-pro"
        if full_model_str.startswith("gemini:"):
            real_model = full_model_str[len("gemini:"):]
            if not self.gemini_api_key:
                raise UserError(
                    "No Gemini API key found. Either set GEMINI_API_KEY in .env or pass gemini_api_key="
                )
            return OpenAIModel(
                real_model,
                base_url="https://api.gemini.com",
                api_key=self.gemini_api_key,
            )

        # Otherwise, handle as "normal" (including known 'openai:gpt-4o-mini', 'groq:...', 'vertexai:...', etc.)
        # or let pydantic_ai.models.infer_model handle it.
        # If it's an openai prefix, we use the openai_api_key if it's an OpenAIModel.
        # Otherwise, let the relevant provider's environment usage happen under the hood.
        if isinstance(full_model_str, (KnownModelName, str)):
            # If "openai:" is detected, or user uses a name that leads to OpenAIModel, we pass openai_api_key if set.
            # This is a fallback. We'll attempt to infer the model:
            inst = infer_model(full_model_str)
            if isinstance(inst, OpenAIModel) and self.openai_api_key:
                # override the default if we have an explicit openai_api_key
                inst.api_key = self.openai_api_key
            return inst

        # If it's already a pydantic_ai Model, just return it.
        if isinstance(full_model_str, Model):
            return full_model_str

        # If none of the above, raise an error
        raise UserError(f"Unrecognized model reference: {full_model_str}")

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: Union[str, KnownModelName, Model],
        response_type: Type[T],
        *,
        system_message: Union[str, Sequence[str]] = (),
        batch_size: int = 1000,
        batch_mode: bool = False,
        retries: int = 1,
    ) -> UnifiedResponse[Union[T, List[T], BatchResult]]:
        """
        Main entry point for processing user prompts with typed responses.
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

                # If user gave a raw string for the model, let's maybe add a prefix if needed.
                if isinstance(model, str):
                    # Potentially map short model names => "openai:gpt-4o-mini" etc.
                    model_mapped = self._map_short_model_name(model)
                    model_instance = self._build_model_instance(model_mapped)
                elif isinstance(model, KnownModelName):
                    # KnownModelName => pass to build
                    model_instance = self._build_model_instance(model)
                elif isinstance(model, Model):
                    # Already a pydantic_ai model, just trust it
                    model_instance = model
                else:
                    # Fallback: treat it as a string
                    raise UserError(f"Invalid model parameter: {model}")

                agent = Agent(
                    model_instance,
                    result_type=response_type,
                    system_prompt=system_message,
                    retries=retries,
                )

                if batch_mode:
                    # Batch mode is only supported by OpenAI (OpenAIModel)
                    if not isinstance(model_instance, OpenAIModel):
                        raise UserError(
                            "Batch API mode is only supported for OpenAI models."
                        )
                    batch_result = await self._process_batch(
                        agent, prompts, response_type
                    )
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
                with logfire.span(
                    "error_handling", error=str(e), error_type="user_error"
                ):
                    return UnifiedResponse(
                        success=False,
                        error=error_msg,
                        original_prompt=original_prompt_for_error,
                    )
            except Exception as e:
                full_trace = traceback.format_exc()
                error_msg = f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                with logfire.span(
                    "error_handling", error=str(e), error_type="unexpected_error"
                ):
                    return UnifiedResponse(
                        success=False,
                        error=error_msg,
                        original_prompt=original_prompt_for_error,
                    )

    async def _process_single(self, agent: Agent, prompt: str) -> T:
        """
        Process a single prompt with optional rate limiting (KISS).
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
        Process multiple prompts, chunked by `batch_size` to avoid sending too large requests.
        Uses asyncio.gather for parallelism (Single Responsibility).
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
        Specialized method for the OpenAI Batch API workflow. Writes JSONL, uploads,
        polls for completion, and returns a BatchResult with typed responses.
        (Single Responsibility: only handles batch logic.)
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

            # This uses the OpenAI python client under the hood
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
                        content = data["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                        r = response_type.construct()
                        if "content" in response_type.model_fields:
                            setattr(r, "content", content)
                        if "confidence" in response_type.model_fields:
                            setattr(r, "confidence", 0.95)
                        results.append({"prompt": prompt, "response": r})
                    except Exception as e:
                        full_trace = traceback.format_exc()
                        error_msg = (
                            f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                        )
                        results.append({"prompt": prompt, "error": error_msg})

            return BatchResult(metadata=metadata, results=results)


# ----------------------------------------------------------------------------
# Testing / Demonstration
# ----------------------------------------------------------------------------

async def run_tests():
    """
    Demonstrates usage of UnifiedLLMHandler with single, multiple,
    and batch prompts, plus usage of new model prefixes like openrouter:, deepseek:, etc.

    PREREQUISITE:
      - Ensure environment variables or constructor arguments are set for each provider you plan to use.
    """
    # Example instantiation with explicit keys (any omitted keys fall back to environment)
    handler = UnifiedLLMHandler(
        requests_per_minute=2000,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Test 1: Single prompt
    single_result = await handler.process(
        "Explain quantum computing in simple terms",
        "gpt-4o-mini",  # normal usage => openai prefix
        SimpleResponse,
    )
    print("\nTest 1 (Single Prompt):")
    print("Success?", single_result.success)
    print(
        "Data:",
        single_result.data.model_dump(mode="python")
        if single_result.success and single_result.data
        else single_result.error,
    )

    # Test 2: Multiple prompts
    math_questions = ["What is 127+345?", "Calculate 15% of 2500", "What is sqrt(169)?"]
    multi_result = await handler.process(
        math_questions,
        "gemini-1.5-flash-8b",
        MathResponse,
        system_message="You are a precise mathematical assistant. Always show your reasoning.",
    )
    print("\nTest 2 (Multiple Prompts):")
    print("Success?", multi_result.success)
    if multi_result.success and multi_result.data:
        if isinstance(multi_result.data, list):
            for d in multi_result.data:
                print(d.model_dump(mode="python"))
        else:
            print(multi_result.data.model_dump(mode="python"))
    else:
        print(multi_result.error)

    # Test 3: Parallel processing with multiple prompts
    person_prompts = [
        "Describe a software engineer in Silicon Valley",
        "Describe a chef in a Michelin star restaurant",
    ]
    person_result = await handler.process(
        person_prompts, "gpt-4o-mini", PersonResponse, batch_size=2
    )
    print("\nTest 3 (Parallel Processing):")
    print("Success?", person_result.success)
    if person_result.success and isinstance(person_result.data, list):
        for d in person_result.data:
            print(d.model_dump(mode="python"))
    else:
        print(person_result.error)

    # Test 4: Batch API processing with invalid model
    # (gemini-1.5-flash-8b is not an OpenAI model => error)
    simple_prompts = [f"Write a short story about number {i}" for i in range(3)]
    batch_result = await handler.process(
        simple_prompts,
        "gemini-1.5-flash-8b",
        SimpleResponse,
        batch_mode=True,
    )
    print("\nTest 4 (Batch API with invalid model):")
    print("Success?", batch_result.success)
    print(
        "Output:",
        batch_result.error
        if not batch_result.success
        else batch_result.data.model_dump(mode="python"),
    )

    # Test 5: Invalid model
    invalid_model_result = await handler.process(
        "Test prompt", "invalid-model", SimpleResponse
    )
    print("\nTest 5 (Invalid Model):")
    print("Success?", invalid_model_result.success)
    print(
        "Output:",
        invalid_model_result.error
        if not invalid_model_result.success
        else invalid_model_result.data.model_dump(mode="python"),
    )

    # Test 6: Invalid prompt (None)
    invalid_prompt_result = await handler.process(None, "gpt-4o-mini", SimpleResponse)
    print("\nTest 6 (Invalid Prompt):")
    print("Success?", invalid_prompt_result.success)
    print(
        "Output:",
        invalid_prompt_result.error
        if not invalid_prompt_result.success
        else invalid_prompt_result.data.model_dump(mode="python"),
    )

    # Test 7: OpenRouter usage example
    # e.g. "openrouter:anthropic/claude-3.5-sonnet"
    # Note: Must have OPENROUTER_API_KEY or pass openrouter_api_key= in constructor
    openrouter_result = await handler.process(
        "Hello from OpenRouter! Please summarize the concept of Pydantic AI structured outputs.",
        "openrouter:anthropic/claude-3.5-sonnet",
        SimpleResponse,
    )
    print("\nTest 7 (OpenRouter Usage):")
    print("Success?", openrouter_result.success)
    if openrouter_result.success and openrouter_result.data:
        if isinstance(openrouter_result.data, SimpleResponse):
            print(openrouter_result.data.model_dump(mode="python"))
        else:
            print(openrouter_result.data)
    else:
        print(openrouter_result.error)

    # Test 8: DeepSeek usage example
    # e.g. "deepseek:deepseek-chat"
    # Note: Must have DEEPSEEK_API_KEY or pass deepseek_api_key= in constructor
    deepseek_result = await handler.process(
        "Hello from DeepSeek! Please list 3 unique features of your platform.",
        "deepseek:deepseek-chat",
        SimpleResponse,
    )
    print("\nTest 8 (DeepSeek Usage):")
    print("Success?", deepseek_result.success)
    if deepseek_result.success and deepseek_result.data:
        if isinstance(deepseek_result.data, SimpleResponse):
            print(deepseek_result.data.model_dump(mode="python"))
        else:
            print(deepseek_result.data)
    else:
        print(deepseek_result.error)

    # Test 9: Anthropic usage example (Requires pydantic-ai[anthropic] installed)
    # e.g. "anthropic:claude-2"
    if AnthropicModel is not None:
        anthro_result = await handler.process(
            "Explain YAGNI principle in software development.",
            "anthropic:claude-2",
            SimpleResponse,
        )
        print("\nTest 9 (Anthropic Usage):")
        print("Success?", anthro_result.success)
        if anthro_result.success and anthro_result.data:
            if isinstance(anthro_result.data, SimpleResponse):
                print(anthro_result.data.model_dump(mode="python"))
            else:
                print(anthro_result.data)
        else:
            print(anthro_result.error)
    else:
        print("\nTest 9 (Anthropic Usage): Skipped because AnthropicModel not installed.")


if __name__ == "__main__":
    asyncio.run(run_tests())
