# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import create_model

from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.registry_handler import RegistryHandlerBaseConfig
from nat.settings.global_settings import GlobalSettings

logger = logging.getLogger(__name__)


def configure_registry_channel(config_type: RegistryHandlerBaseConfig, channel_name: str) -> None:
    """Perform channel updates, gathering input from user and validatinig against the global settings data model.

    Args:
        config_type (RegistryHandlerBaseConfig): The registry handler configuration object to ensure valid channel
        settings
        channel_name (str): The name to use to reference the remote registry channel.
    """

    settings = GlobalSettings.get()

    channel_registry_pre = {}

    for field, info in config_type.model_fields.items():

        if (field == "type"):
            continue

        while (True):
            human_prompt = " ".join(field.title().split("_"))
            user_input = input(f"{human_prompt}: ")
            model_fields = {}
            model_fields[field] = (info.annotation, ...)
            DynamicFieldModel = create_model("DynamicFieldModel", **model_fields)
            dynamic_inputs = {field: user_input}

            try:
                validated_field_model = DynamicFieldModel(**dynamic_inputs)
                channel_registry_pre[field] = getattr(validated_field_model, field)
                break
            except Exception as e:
                logger.exception(e)
                logger.warning("Invalid '%s' input, input must be of type %s.", field, info.annotation)

    validated_model = config_type(**channel_registry_pre)
    settings_dict = settings.model_dump(serialize_as_any=True, by_alias=True)
    settings_dict["channels"] = {**settings_dict["channels"], **{channel_name: validated_model}}

    settings.update_settings(config_obj=settings_dict)


def add_channel_interative(channel_type: str) -> None:
    """Add a remote registry channel to publish/search/pull NAT plugin packages.

    Args:
        channel_type (str): They type of channel to configure.
    """

    settings = GlobalSettings.get()
    registry = GlobalTypeRegistry.get()

    try:
        ChannelConfigType = registry.get_registered_channel_info_by_channel_type(channel_type=channel_type).config_type
    except Exception as e:
        logger.exception("Invalid channel type: %s", e)
        return

    while (True):
        channel_name = input("Channel Name: ").strip()
        if len(channel_name) < 1:
            logger.warning("Invalid channel name, cannot be empty or whitespace.")
        if (channel_name in settings.channels):
            logger.warning("Channel name '%s' already exists, choose a different name.", channel_name)
        else:
            settings.channels[channel_name] = {}
            break

    ChannelConfigType = registry.get_registered_channel_info_by_channel_type(channel_type=channel_type).config_type

    configure_registry_channel(config_type=ChannelConfigType, channel_name=channel_name)


def get_existing_channel_interactive(channel_name: str) -> tuple[str, bool]:
    """Retrieve an existing channel by configured name.

    Args:
        channel_name (str): The name to use to reference the remote registry channel.

    Returns:
        tuple[str, bool]: A tuple containing the retrieved channel name and a boolean representing a
            valid match was or was not successful.
    """

    settings = GlobalSettings.get()
    valid_channel = False
    remote_channels = settings.channels

    if (len(remote_channels) == 0):
        logger.warning("No are configured channels to remove.")
        return channel_name, valid_channel

    while (not valid_channel):

        if (channel_name not in remote_channels):
            logger.warning("Channel name '%s' does not exist, choose a name from %s",
                           channel_name,
                           settings.channel_names)
            channel_name = input("Channel Name: ").strip()
            continue

        valid_channel = True

    return channel_name, valid_channel


def remove_channel(channel_name: str) -> None:
    """Remove a configured registry channel from the global settings.

    Args:
        channel_name (str): The name to use to reference the remote registry channel.
    """

    settings = GlobalSettings.get()
    settings_dict = settings.model_dump(serialize_as_any=True, by_alias=True).copy()
    settings_dict["channels"].pop(channel_name)
    settings.update_settings(config_obj=settings_dict)


def remove_channel_interactive(channel_name: str) -> None:
    channel_name, valid_channel = get_existing_channel_interactive(channel_name=channel_name)
    if (not valid_channel):
        return
    remove_channel(channel_name=channel_name)


def match_valid_channel(channel_name: str) -> None:
    """Performs a match by registry channel to perform a channel configuration update.

    Args:
        channel_name (str): The name to use to reference the remote registry channel.
    """

    settings = GlobalSettings.get()
    registry = GlobalTypeRegistry.get()

    if len(settings.channel_names) == 0:
        logger.warning("No channels have been configured, first add a channel.")
        return

    if (channel_name not in settings.channel_names):
        logger.warning("Provided channel has not yet been configured, choose a different name "
                       "from %s .",
                       settings.channel_names)
        while (True):
            channel_name = input("Channel Name: ").strip()
            if len(channel_name) < 1:
                logger.warning("Invalid channel name, cannot be empty or whitespace.")
            if (channel_name in settings.channel_names):
                logger.warning("Channel name '%s' already exists, choose a different name.", channel_name)
            else:
                settings.channels[channel_name] = {}
                break

    channals_settings = settings.channels
    channel_settings = channals_settings.get(channel_name)
    ChannelConfigType = registry.get_registered_channel_info_by_channel_type(
        channel_type=channel_settings.static_type()).config_type

    configure_registry_channel(config_type=ChannelConfigType, channel_name=channel_name)


def update_channel_interactive(channel_name: str):
    """Launch an interactive session to  update a configured channels settings.

    Args:
        channel_name (str): The name to use to reference the remote registry channel.
    """

    match_valid_channel(channel_name=channel_name)
