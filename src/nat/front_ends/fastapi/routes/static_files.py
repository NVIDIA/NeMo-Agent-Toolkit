# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Static file object store endpoints (/static/...).

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


async def register_static_file_routes(worker: FastApiFrontEndPluginWorker, app: FastAPI,
                                      builder: WorkflowBuilder) -> None:

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

        filename = file_path.split("/")[-1]

        async def reader():
            yield file_data.data

        return StreamingResponse(reader(),
                                 media_type=file_data.content_type,
                                 headers={"Content-Disposition": f"attachment; filename={filename}"})

    async def delete_static_file(file_path: str):
        try:
            await object_store_client.delete_object(file_path)
        except NoSuchKeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return Response(status_code=204)

    # Add the static files route to the FastAPI app
    worker._register_api_route(app,
                               path="/static/{file_path:path}",
                               endpoint=add_static_file,
                               methods=["POST"],
                               description="Upload a static file to the object store")

    worker._register_api_route(app,
                               path="/static/{file_path:path}",
                               endpoint=upsert_static_file,
                               methods=["PUT"],
                               description="Upsert a static file to the object store")

    worker._register_api_route(app,
                               path="/static/{file_path:path}",
                               endpoint=get_static_file,
                               methods=["GET"],
                               description="Get a static file from the object store")

    worker._register_api_route(
        app,
        path="/static/{file_path:path}",
        endpoint=delete_static_file,
        methods=["DELETE"],
        description="Delete a static file from the object store",
    )
