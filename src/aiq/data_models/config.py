# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import typing

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import ValidatorFunctionWrapHandler
from pydantic import field_validator

from aiq.data_models.evaluate import EvalConfig
from aiq.data_models.front_end import FrontEndBaseConfig
from aiq.data_models.function import EmptyFunctionConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.its_strategy import ITSStrategyBaseConfig
from aiq.data_models.logging import LoggingBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

from .common import HashableBaseModel
from .common import TypedBaseModel
from .embedder import EmbedderBaseConfig
from .llm import LLMBaseConfig
from .memory import MemoryBaseConfig
from .object_store import ObjectStoreBaseConfig
from .retriever import RetrieverBaseConfig

logger = logging.getLogger(__name__)


def _process_validation_error(err: ValidationError, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):
    from aiq.cli.type_registry import GlobalTypeRegistry  # pylint: disable=cyclic-import

    new_errors = []
    logged_once = False
    needs_reraise = False
    for e in err.errors():

        error_type = e['type']
        if error_type == 'union_tag_invalid' and "ctx" in e and not logged_once:
            requested_type = e["ctx"]["tag"]

            if (info.field_name in ('workflow', 'functions')):
                registered_keys = GlobalTypeRegistry.get().get_registered_functions()
            elif (info.field_name == "llms"):
                registered_keys = GlobalTypeRegistry.get().get_registered_llm_providers()
            elif (info.field_name == "embedders"):
                registered_keys = GlobalTypeRegistry.get().get_registered_embedder_providers()
            elif (info.field_name == "memory"):
                registered_keys = GlobalTypeRegistry.get().get_registered_memorys()
            elif (info.field_name == "object_stores"):
                registered_keys = GlobalTypeRegistry.get().get_registered_object_stores()
            elif (info.field_name == "retrievers"):
                registered_keys = GlobalTypeRegistry.get().get_registered_retriever_providers()
            elif (info.field_name == "tracing"):
                registered_keys = GlobalTypeRegistry.get().get_registered_telemetry_exporters()
            elif (info.field_name == "logging"):
                registered_keys = GlobalTypeRegistry.get().get_registered_logging_method()
            elif (info.field_name == "evaluators"):
                registered_keys = GlobalTypeRegistry.get().get_registered_evaluators()
            elif (info.field_name == "front_ends"):
                registered_keys = GlobalTypeRegistry.get().get_registered_front_ends()
            elif (info.field_name == "its_strategies"):
                registered_keys = GlobalTypeRegistry.get().get_registered_its_strategies()

            else:
                assert False, f"Unknown field name {info.field_name} in validator"

            # Check and see if the there are multiple full types which match this short type
            matching_keys = [k for k in registered_keys if k.local_name == requested_type]

            assert len(matching_keys) != 1, "Exact match should have been found. Contact developers"

            matching_key_names = [x.full_type for x in matching_keys]
            registered_key_names = [x.full_type for x in registered_keys]

            if (len(matching_keys) == 0):
                # This is a case where the requested type is not found. Show a helpful message about what is
                # available
                logger.error(("Requested %s type `%s` not found. "
                              "Have you ensured the necessary package has been installed with `uv pip install`?"
                              "\nAvailable %s names:\n - %s\n"),
                             info.field_name,
                             requested_type,
                             info.field_name,
                             '\n - '.join(registered_key_names))
            else:
                # This is a case where the requested type is ambiguous.
                logger.error(("Requested %s type `%s` is ambiguous. "
                              "Matched multiple %s by their local name: %s. "
                              "Please use the fully qualified %s name."
                              "\nAvailable %s names:\n - %s\n"),
                             info.field_name,
                             requested_type,
                             info.field_name,
                             matching_key_names,
                             info.field_name,
                             info.field_name,
                             '\n - '.join(registered_key_names))

            # Only show one error
            logged_once = True

        elif error_type == 'missing':
            location = e["loc"]
            if len(location) > 1:  # remove the _type field from the location
                e['loc'] = (location[0], ) + location[2:]
                needs_reraise = True

        new_errors.append(e)

    if needs_reraise:
        raise ValidationError.from_exception_data(title=err.title, line_errors=new_errors)


class TelemetryConfig(BaseModel):

    logging: dict[str, LoggingBaseConfig] = {}
    tracing: dict[str, TelemetryExporterBaseConfig] = {}

    @field_validator("logging", "tracing", mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:
            _process_validation_error(err, handler, info)
            raise

    @classmethod
    def rebuild_annotations(cls):

        from aiq.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        TracingAnnotation = dict[str,
                                 typing.Annotated[type_registry.compute_annotation(TelemetryExporterBaseConfig),
                                                  Discriminator(TypedBaseModel.discriminator)]]

        LoggingAnnotation = dict[str,
                                 typing.Annotated[type_registry.compute_annotation(LoggingBaseConfig),
                                                  Discriminator(TypedBaseModel.discriminator)]]

        should_rebuild = False

        tracing_field = cls.model_fields.get("tracing")
        if tracing_field is not None and tracing_field.annotation != TracingAnnotation:
            tracing_field.annotation = TracingAnnotation
            should_rebuild = True

        logging_field = cls.model_fields.get("logging")
        if logging_field is not None and logging_field.annotation != LoggingAnnotation:
            logging_field.annotation = LoggingAnnotation
            should_rebuild = True

        if (should_rebuild):
            return cls.model_rebuild(force=True)

        return False


class GeneralConfig(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    use_uvloop: bool = True
    """
    Whether to use uvloop for the event loop. This can provide a significant speedup in some cases. Disable to provide
    better error messages when debugging.
    """

    telemetry: TelemetryConfig = TelemetryConfig()

    # FrontEnd Configuration
    front_end: FrontEndBaseConfig = FastApiFrontEndConfig()

    @field_validator("front_end", mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:
            _process_validation_error(err, handler, info)
            raise

    @classmethod
    def rebuild_annotations(cls):

        from aiq.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        FrontEndAnnotation = typing.Annotated[type_registry.compute_annotation(FrontEndBaseConfig),
                                              Discriminator(TypedBaseModel.discriminator)]

        should_rebuild = False

        front_end_field = cls.model_fields.get("front_end")
        if front_end_field is not None and front_end_field.annotation != FrontEndAnnotation:
            front_end_field.annotation = FrontEndAnnotation
            should_rebuild = True

        if (TelemetryConfig.rebuild_annotations()):
            should_rebuild = True

        if (should_rebuild):
            return cls.model_rebuild(force=True)

        return False


class AIQConfig(HashableBaseModel):

    model_config = ConfigDict(extra="forbid")

    # Global Options
    general: GeneralConfig = GeneralConfig()

    # Functions Configuration
    functions: dict[str, FunctionBaseConfig] = {}

    # LLMs Configuration
    llms: dict[str, LLMBaseConfig] = {}

    # Embedders Configuration
    embedders: dict[str, EmbedderBaseConfig] = {}

    # Memory Configuration
    memory: dict[str, MemoryBaseConfig] = {}

    # Object Stores Configuration
    object_stores: dict[str, ObjectStoreBaseConfig] = {}

    # Retriever Configuration
    retrievers: dict[str, RetrieverBaseConfig] = {}

    # ITS Strategies
    its_strategies: dict[str, ITSStrategyBaseConfig] = {}

    # Workflow Configuration
    workflow: FunctionBaseConfig = EmptyFunctionConfig()

    # Evaluation Options
    eval: EvalConfig = EvalConfig()

    def print_summary(self, stream: typing.TextIO = sys.stdout):
        """Print a summary of the configuration"""

        stream.write("\nConfiguration Summary:\n")
        stream.write("-" * 20 + "\n")
        if self.workflow:
            stream.write(f"Workflow Type: {self.workflow.type}\n")

        stream.write(f"Number of Functions: {len(self.functions)}\n")
        stream.write(f"Number of LLMs: {len(self.llms)}\n")
        stream.write(f"Number of Embedders: {len(self.embedders)}\n")
        stream.write(f"Number of Memory: {len(self.memory)}\n")
        stream.write(f"Number of Object Stores: {len(self.object_stores)}\n")
        stream.write(f"Number of Retrievers: {len(self.retrievers)}\n")
        stream.write(f"Number of ITS Strategies: {len(self.its_strategies)}\n")

    @field_validator("functions",
                     "llms",
                     "embedders",
                     "memory",
                     "retrievers",
                     "workflow",
                     "its_strategies",
                     mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:
            _process_validation_error(err, handler, info)
            raise

    @classmethod
    def rebuild_annotations(cls):

        from aiq.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        LLMsAnnotation = dict[str,
                              typing.Annotated[type_registry.compute_annotation(LLMBaseConfig),
                                               Discriminator(TypedBaseModel.discriminator)]]

        EmbeddersAnnotation = dict[str,
                                   typing.Annotated[type_registry.compute_annotation(EmbedderBaseConfig),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        FunctionsAnnotation = dict[str,
                                   typing.Annotated[type_registry.compute_annotation(FunctionBaseConfig, ),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        MemoryAnnotation = dict[str,
                                typing.Annotated[type_registry.compute_annotation(MemoryBaseConfig),
                                                 Discriminator(TypedBaseModel.discriminator)]]

        ObjectStoreAnnotation = dict[str,
                                     typing.Annotated[type_registry.compute_annotation(ObjectStoreBaseConfig),
                                                      Discriminator(TypedBaseModel.discriminator)]]

        RetrieverAnnotation = dict[str,
                                   typing.Annotated[type_registry.compute_annotation(RetrieverBaseConfig),
                                                    Discriminator(TypedBaseModel.discriminator)]]

        ITSStrategyAnnotation = dict[str,
                                     typing.Annotated[type_registry.compute_annotation(ITSStrategyBaseConfig),
                                                      Discriminator(TypedBaseModel.discriminator)]]

        WorkflowAnnotation = typing.Annotated[type_registry.compute_annotation(FunctionBaseConfig),
                                              Discriminator(TypedBaseModel.discriminator)]

        should_rebuild = False

        llms_field = cls.model_fields.get("llms")
        if llms_field is not None and llms_field.annotation != LLMsAnnotation:
            llms_field.annotation = LLMsAnnotation
            should_rebuild = True

        embedders_field = cls.model_fields.get("embedders")
        if embedders_field is not None and embedders_field.annotation != EmbeddersAnnotation:
            embedders_field.annotation = EmbeddersAnnotation
            should_rebuild = True

        functions_field = cls.model_fields.get("functions")
        if functions_field is not None and functions_field.annotation != FunctionsAnnotation:
            functions_field.annotation = FunctionsAnnotation
            should_rebuild = True

        memory_field = cls.model_fields.get("memory")
        if memory_field is not None and memory_field.annotation != MemoryAnnotation:
            memory_field.annotation = MemoryAnnotation
            should_rebuild = True

        object_stores_field = cls.model_fields.get("object_stores")
        if object_stores_field is not None and object_stores_field.annotation != ObjectStoreAnnotation:
            object_stores_field.annotation = ObjectStoreAnnotation
            should_rebuild = True

        retrievers_field = cls.model_fields.get("retrievers")
        if retrievers_field is not None and retrievers_field.annotation != RetrieverAnnotation:
            retrievers_field.annotation = RetrieverAnnotation
            should_rebuild = True

        its_strategies_field = cls.model_fields.get("its_strategies")
        if its_strategies_field is not None and its_strategies_field.annotation != ITSStrategyAnnotation:
            its_strategies_field.annotation = ITSStrategyAnnotation
            should_rebuild = True

        workflow_field = cls.model_fields.get("workflow")
        if workflow_field is not None and workflow_field.annotation != WorkflowAnnotation:
            workflow_field.annotation = WorkflowAnnotation
            should_rebuild = True

        if (GeneralConfig.rebuild_annotations()):
            should_rebuild = True

        if (EvalConfig.rebuild_annotations()):
            should_rebuild = True

        if (should_rebuild):
            return cls.model_rebuild(force=True)

        return False
