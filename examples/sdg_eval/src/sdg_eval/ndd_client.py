# flake8: noqa: E501, BLE001
import logging
from typing import Dict, Any, List, Optional

from pydantic import Field, BaseModel
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class NDDClientConfig(FunctionBaseConfig, name="ndd_client"):
    """Configuration for the NeMo Data Designer client tool."""
    model_name: str = Field(default="gpt-3.5-turbo", description="Model to use for data generation")
    num_samples: int = Field(default=10, description="Number of synthetic samples to generate")
    temperature: float = Field(default=0.7, description="Temperature for data generation")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class DataGenerationRequest(BaseModel):
    """Request model for data generation."""
    prompt: str = Field(description="Prompt for data generation")
    schema: Optional[Dict[str, Any]] = Field(default=None, description="Schema to follow for generated data")
    examples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Example data points")
    instructions: Optional[str] = Field(default=None, description="Additional instructions for generation")


class DataGenerationResponse(BaseModel):
    """Response model for data generation."""
    generated_data: List[Dict[str, Any]] = Field(description="Generated synthetic data")
    metadata: Dict[str, Any] = Field(description="Generation metadata")
    success: bool = Field(description="Whether generation was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if generation failed")


@register_function(config_type=NDDClientConfig)
async def ndd_client_function(config: NDDClientConfig, _builder: Builder):
    """NeMo Data Designer client for synthetic data generation."""

    async def generate_synthetic_data(request: DataGenerationRequest) -> DataGenerationResponse:
        """Generate synthetic data using NeMo Data Designer patterns."""
        try:
            logger.info("Generating %d synthetic samples", config.num_samples)

            generated_samples = []

            # Simulate synthetic data generation based on the prompt and schema
            for i in range(config.num_samples):
                sample = await _generate_single_sample(
                    prompt=request.prompt,
                    schema=request.schema,
                    examples=request.examples,
                    instructions=request.instructions,
                    seed=config.seed + i if config.seed else None
                )
                generated_samples.append(sample)

            metadata = {
                "model_name": config.model_name,
                "num_samples": config.num_samples,
                "temperature": config.temperature,
                "seed": config.seed,
                "prompt_length": len(request.prompt),
                "schema_provided": request.schema is not None,
                "examples_provided": request.examples is not None and len(request.examples) > 0
            }

            return DataGenerationResponse(
                generated_data=generated_samples,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error("Data generation failed: %s", str(e))
            return DataGenerationResponse(
                generated_data=[],
                metadata={},
                success=False,
                error_message=str(e)
            )

    async def _generate_single_sample(
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a single synthetic data sample."""

        # Basic synthetic data generation logic
        # In a real implementation, this would use NeMo Data Designer APIs

        sample = {}

        # If schema is provided, generate data following the schema
        if schema and "properties" in schema:
            for field_name, field_spec in schema["properties"].items():
                field_type = field_spec.get("type", "string")

                if field_type == "string":
                    sample[field_name] = f"synthetic_{field_name}_{seed or 'default'}"
                elif field_type == "integer":
                    sample[field_name] = (seed or 42) % 100
                elif field_type == "number":
                    sample[field_name] = (seed or 3.14) * 0.1
                elif field_type == "boolean":
                    sample[field_name] = (seed or 1) % 2 == 0
                elif field_type == "array":
                    sample[field_name] = [f"item_{i}" for i in range(2)]
                else:
                    sample[field_name] = f"synthetic_value_{seed or 'default'}"

        # If examples are provided, use them as templates
        elif examples and len(examples) > 0:
            template = examples[0]
            for key, value in template.items():
                if isinstance(value, str):
                    sample[key] = f"synthetic_{value}_{seed or 'default'}"
                elif isinstance(value, int):
                    sample[key] = value + (seed or 1)
                elif isinstance(value, float):
                    sample[key] = value * 1.1
                else:
                    sample[key] = value

        # Default generation based on prompt
        else:
            sample = {
                "generated_text": f"Synthetic data based on: {prompt[:50]}...",
                "sample_id": seed or "default",
                "prompt_hash": hash(prompt) % 1000
            }

        # Add generation metadata
        sample["_generation_metadata"] = {
            "prompt_snippet": prompt[:100],
            "instructions": instructions,
            "generated_at": "synthetic_timestamp",
            "seed": seed
        }

        return sample

    async def create_dataset_from_schema(schema: Dict[str, Any], _num_samples: int = None) -> Dict[str, Any]:
        """Create a complete dataset from a schema definition."""
        try:
            request = DataGenerationRequest(
                prompt="Generate data following the provided schema",
                schema=schema,
                instructions="Create diverse, realistic synthetic data"
            )

            response = await generate_synthetic_data(request)

            if response.success:
                dataset = {
                    "data": response.generated_data,
                    "schema": schema,
                    "metadata": {
                        **response.metadata,
                        "dataset_size": len(response.generated_data),
                        "schema_fields": list(schema.get("properties", {}).keys())
                    }
                }
                return {"dataset": dataset, "success": True}
            else:
                return {"error": response.error_message, "success": False}

        except Exception as e:
            logger.error("Dataset creation failed: %s", str(e))
            return {"error": str(e), "success": False}

    async def augment_existing_data(existing_data: List[Dict[str, Any]], augmentation_factor: int = 2) -> Dict[str, Any]:
        """Augment existing data by generating similar synthetic samples."""
        try:
            if not existing_data:
                return {"error": "No existing data provided", "success": False}

            # Use existing data as examples
            request = DataGenerationRequest(
                prompt="Generate similar data to the provided examples",
                examples=existing_data[:5],  # Use first 5 as examples
                instructions="Create variations that maintain the same structure and style"
            )

            # Generate augmented data
            total_samples = len(existing_data) * augmentation_factor
            original_num_samples = config.num_samples
            config.num_samples = total_samples

            response = await generate_synthetic_data(request)
            config.num_samples = original_num_samples  # Reset

            if response.success:
                augmented_dataset = {
                    "original_data": existing_data,
                    "synthetic_data": response.generated_data,
                    "combined_data": existing_data + response.generated_data,
                    "metadata": {
                        **response.metadata,
                        "original_size": len(existing_data),
                        "synthetic_size": len(response.generated_data),
                        "total_size": len(existing_data) + len(response.generated_data),
                        "augmentation_factor": augmentation_factor
                    }
                }
                return {"augmented_dataset": augmented_dataset, "success": True}
            else:
                return {"error": response.error_message, "success": False}

        except Exception as e:
            logger.error("Data augmentation failed: %s", str(e))
            return {"error": str(e), "success": False}

    try:
        yield FunctionInfo.create(
            single_fn=generate_synthetic_data,
            description="Generate synthetic data using NeMo Data Designer patterns",
        )

        yield FunctionInfo.create(
            single_fn=create_dataset_from_schema,
            description="Create a complete synthetic dataset from a schema definition",
        )

        yield FunctionInfo.create(
            single_fn=augment_existing_data,
            description="Augment existing data with synthetic samples",
        )

    except GeneratorExit:
        logger.warning("NDD Client function exited early!")
    finally:
        logger.info("Cleaning up NDD Client workflow.")
