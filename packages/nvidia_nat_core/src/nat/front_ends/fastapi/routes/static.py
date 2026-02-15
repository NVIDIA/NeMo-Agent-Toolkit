# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Static file route registration."""

import logging
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.models import ObjectStoreItem

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

logger = logging.getLogger(__name__)


async def add_static_files_route(worker: "FastApiFrontEndPluginWorker", app: FastAPI, builder: WorkflowBuilder):
    """Add static file CRUD routes when object-store support is configured."""
    if not worker.front_end_config.object_store:
        logger.debug("No object store configured, skipping static files route")
        return

    object_store_client = await builder.get_object_store_client(worker.front_end_config.object_store)

    def sanitize_path(path: str) -> str:
        sanitized_path = os.path.normpath(path.strip("/"))
        if sanitized_path == ".":
            raise HTTPException(status_code=400, detail="Invalid file path.")

        filename = os.path.basename(sanitized_path)
        if not filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty.")

        return sanitized_path

    # Upload static files to the object store; if key is present, it will fail with 409 Conflict
    async def add_static_file(file_path: str, file: UploadFile):
        sanitized_file_path = sanitize_path(file_path)
        file_data = await file.read()

        try:
            await object_store_client.put_object(sanitized_file_path,
                                                 ObjectStoreItem(data=file_data, content_type=file.content_type))
        except KeyAlreadyExistsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        return {"filename": sanitized_file_path}

    # Upsert static files to the object store; if key is present, it will overwrite the file
    async def upsert_static_file(file_path: str, file: UploadFile):
        sanitized_file_path = sanitize_path(file_path)
        file_data = await file.read()

        await object_store_client.upsert_object(sanitized_file_path,
                                                ObjectStoreItem(data=file_data, content_type=file.content_type))
        return {"filename": sanitized_file_path}

    # Get static files from the object store
    async def get_static_file(file_path: str):
        try:
            file_data = await object_store_client.get_object(file_path)
        except NoSuchKeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        filename = file_path.rsplit("/", maxsplit=1)[-1]

        # Sanitize filename for Content-Disposition header (RFC 6266).
        # The ASCII fallback uses only safe characters; the filename* parameter
        # carries the full UTF-8 percent-encoded name.
        ascii_safe = re.sub(r'[^\w.\-]', '_', filename)
        utf8_encoded = quote(filename, safe='')
        content_disposition = (f'attachment; filename="{ascii_safe}"; '
                               f"filename*=UTF-8''{utf8_encoded}")

        async def reader():
            yield file_data.data

        return StreamingResponse(reader(),
                                 media_type=file_data.content_type,
                                 headers={"Content-Disposition": content_disposition})

    async def delete_static_file(file_path: str):
        try:
            await object_store_client.delete_object(file_path)
        except NoSuchKeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return Response(status_code=204)

    app.add_api_route(
        path="/static/{file_path:path}",
        endpoint=add_static_file,
        methods=["POST"],
        description="Upload a static file to the object store",
    )
    app.add_api_route(
        path="/static/{file_path:path}",
        endpoint=upsert_static_file,
        methods=["PUT"],
        description="Upsert a static file to the object store",
    )
    app.add_api_route(
        path="/static/{file_path:path}",
        endpoint=get_static_file,
        methods=["GET"],
        description="Get a static file from the object store",
    )
    app.add_api_route(
        path="/static/{file_path:path}",
        endpoint=delete_static_file,
        methods=["DELETE"],
        description="Delete a static file from the object store",
    )
