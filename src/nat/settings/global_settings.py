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

import json
import logging
import os
import typing
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy

from platformdirs import user_config_dir
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Tag
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import ValidatorFunctionWrapHandler
from pydantic import field_validator

from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import RegisteredInfo
from nat.data_models.common import HashableBaseModel
from nat.data_models.common import TypedBaseModel
from nat.data_models.common import TypedBaseModelT
from nat.data_models.registry_handler import RegistryHandlerBaseConfig

logger = logging.getLogger(__name__)


class Settings(HashableBaseModel):

    model_config = ConfigDict(extra="forbid")

    # Registry Handeler Configuration
    channels: dict[str, RegistryHandlerBaseConfig] = {}

    _configuration_directory: typing.ClassVar[str]
    _settings_changed_hooks: typing.ClassVar[list[Callable[[], None]]] = []
    _settings_changed_hooks_active: bool = True

    @field_validator("channels", mode="wrap")
    @classmethod
    def validate_components(cls, value: typing.Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo):

        try:
            return handler(value)
        except ValidationError as err:

            for e in err.errors():
                if e['type'] == 'union_tag_invalid' and len(e['loc']) > 0:
                    requested_type = e['loc'][0]

                    if (info.field_name == "channels"):
                        registered_keys = GlobalTypeRegistry.get().get_registered_registry_handlers()
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
                        raise ValueError(
                            f"Requested {info.field_name} type `{requested_type}` not found. "
                            "Have you ensured the necessary package has been installed with `uv pip install`?"
                            "\nAvailable {} names:\n - {}".format(info.field_name,
                                                                  '\n - '.join(registered_key_names))) from err

                    # This is a case where the requested type is ambiguous.
                    raise ValueError(f"Requested {info.field_name} type `{requested_type}` is ambiguous. " +
                                     f"Matched multiple {info.field_name} by their local name: {matching_key_names}. " +
                                     f"Please use the fully qualified {info.field_name} name." +
                                     "\nAvailable {} names:\n - {}".format(info.field_name,
                                                                           '\n - '.join(registered_key_names))) from err

            raise

    @classmethod
    def rebuild_annotations(cls):

        def compute_annotation(cls: type[TypedBaseModelT], registrations: list[RegisteredInfo[TypedBaseModelT]]):

            while (len(registrations) < 2):
                registrations.append(RegisteredInfo[TypedBaseModelT](full_type=f"_ignore/{len(registrations)}",
                                                                     config_type=cls))

            short_names: dict[str, int] = {}
            type_list: list[tuple[str, type[TypedBaseModelT]]] = []

            # For all keys in the list, split the key by / and increment the count of the last element
            for key in registrations:
                short_names[key.local_name] = short_names.get(key.local_name, 0) + 1

                type_list.append((key.full_type, key.config_type))

            # Now loop again and if the short name is unique, then create two entries, for the short and full name
            for key in registrations:

                if (short_names[key.local_name] == 1):
                    type_list.append((key.local_name, key.config_type))

            # pylint: disable=consider-alternative-union-syntax
            return typing.Union[tuple(typing.Annotated[x_type, Tag(x_id)] for x_id, x_type in type_list)]

        RegistryHandlerAnnotation = dict[
            str,
            typing.Annotated[compute_annotation(RegistryHandlerBaseConfig,
                                                GlobalTypeRegistry.get().get_registered_registry_handlers()),
                             Discriminator(TypedBaseModel.discriminator)]]

        should_rebuild = False

        channels_field = cls.model_fields.get("channels")
        if channels_field is not None and channels_field.annotation != RegistryHandlerAnnotation:
            channels_field.annotation = RegistryHandlerAnnotation
            should_rebuild = True

        if (should_rebuild):
            cls.model_rebuild(force=True)

    @property
    def channel_names(self) -> list:
        return list(self.channels.keys())

    @property
    def configuration_directory(self) -> str:
        return self._configuration_directory

    @property
    def configuration_file(self) -> str:
        return os.path.join(self.configuration_directory, "config.json")

    @staticmethod
    def from_file():

        configuration_directory = os.getenv("NAT_CONFIG_DIR", user_config_dir(appname="nat"))

        if not os.path.exists(configuration_directory):
            os.makedirs(configuration_directory, exist_ok=True)

        configuration_file = os.path.join(configuration_directory, "config.json")

        file_path = os.path.join(configuration_directory, "config.json")

        if (not os.path.exists(configuration_file)):
            loaded_config = {}
        else:
            with open(file_path, mode="r", encoding="utf-8") as f:
                loaded_config = json.load(f)

        settings = Settings(**loaded_config)
        settings.set_configuration_directory(configuration_directory)
        return settings

    def set_configuration_directory(self, directory: str, remove: bool = False) -> None:
        if (remove):
            if os.path.exists(self.configuration_directory):
                os.rmdir(self.configuration_directory)
        self.__class__._configuration_directory = directory

    def reset_configuration_directory(self, remove: bool = False) -> None:
        if (remove):
            if os.path.exists(self.configuration_directory):
                os.rmdir(self.configuration_directory)
        self._configuration_directory = os.getenv("NAT_CONFIG_DIR", user_config_dir(appname="nat"))

    def _save_settings(self) -> None:

        if not os.path.exists(self.configuration_directory):
            os.mkdir(self.configuration_directory)

        with open(self.configuration_file, mode="w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4, by_alias=True, serialize_as_any=True))

        self._settings_changed()

    def update_settings(self, config_obj: "dict | Settings"):
        self._update_settings(config_obj)

    def _update_settings(self, config_obj: "dict | Settings"):

        if isinstance(config_obj, Settings):
            config_obj = config_obj.model_dump(serialize_as_any=True, by_alias=True)

        self._revalidate(config_dict=config_obj)

        self._save_settings()

    def _revalidate(self, config_dict) -> bool:

        try:
            validated_data = self.__class__(**config_dict)

            for field in validated_data.model_fields:
                match field:
                    case "channels":
                        self.channels = validated_data.channels
                    case _:
                        raise ValueError(f"Encountered invalid model field: {field}")

            return True

        except Exception as e:
            logger.exception("Unable to validate user settings configuration: %s", e, exc_info=True)
            return False

    def print_channel_settings(self, channel_type: str | None = None) -> None:

        import yaml

        remote_channels = self.model_dump(serialize_as_any=True, by_alias=True)

        if (not remote_channels or not remote_channels.get("channels")):
            logger.warning("No configured channels to list.")
            return

        if (channel_type is not None):
            filter_channels = []
            for channel, settings in remote_channels.items():
                if (settings["type"] != channel_type):
                    filter_channels.append(channel)
            for channel in filter_channels:
                del remote_channels[channel]

        if (remote_channels):
            logger.info(yaml.dump(remote_channels, allow_unicode=True, default_flow_style=False))

    def override_settings(self, config_file: str) -> "Settings":

        from nat.utils.io.yaml_tools import yaml_load

        override_settings_dict = yaml_load(config_file)

        settings_dict = self.model_dump()
        updated_settings = {**override_settings_dict, **settings_dict}
        self._update_settings(config_obj=updated_settings)

        return self

    def _settings_changed(self):

        if (not self._settings_changed_hooks_active):
            return

        for hook in self._settings_changed_hooks:
            hook()

    @contextmanager
    def pause_settings_changed_hooks(self):

        self._settings_changed_hooks_active = False

        try:
            yield
        finally:
            self._settings_changed_hooks_active = True

            # Ensure that the registration changed hooks are called
            self._settings_changed()

    def add_settings_changed_hook(self, cb: Callable[[], None]) -> None:

        self._settings_changed_hooks.append(cb)


GlobalTypeRegistry.get().add_registration_changed_hook(lambda: Settings.rebuild_annotations())


class GlobalSettings:

    _global_settings: Settings | None = None

    @staticmethod
    def get() -> Settings:

        if (GlobalSettings._global_settings is None):
            from nat.runtime.loader import PluginTypes
            from nat.runtime.loader import discover_and_register_plugins

            discover_and_register_plugins(PluginTypes.REGISTRY_HANDLER)

            GlobalSettings._global_settings = Settings.from_file()

        return GlobalSettings._global_settings

    @staticmethod
    @contextmanager
    def push():

        saved = GlobalSettings.get()
        settings = deepcopy(saved)

        try:
            GlobalSettings._global_settings = settings

            yield settings
        finally:
            GlobalSettings._global_settings = saved
            GlobalSettings._global_settings._settings_changed()
