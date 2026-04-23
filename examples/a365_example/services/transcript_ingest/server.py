from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp
from aiohttp import web
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
PORT = int(os.environ.get("TRANSCRIPT_INGEST_PORT", "8110"))
DEFAULT_RESOURCES = (
    "communications/onlineMeetings/getAllTranscripts",
)


class GraphClient:
    def __init__(self) -> None:
        self.client_id = _require_env("GRAPH_CLIENT_ID")
        self.client_secret = _require_env("GRAPH_CLIENT_SECRET")
        self.tenant_id = _require_env("GRAPH_TENANT_ID")
        self._token: str | None = None
        self._expires_at: datetime | None = None

    async def _acquire_token(self) -> str:
        if self._token and self._expires_at and datetime.now(UTC) < self._expires_at:
            return self._token

        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": "https://graph.microsoft.com/.default",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"Failed to acquire Graph token: {body}")
                self._token = body["access_token"]
                expires_in = int(body.get("expires_in", 3600))
                self._expires_at = datetime.now(UTC) + timedelta(seconds=max(expires_in - 60, 60))
                return self._token

    async def get_json(self, url_or_path: str) -> dict[str, Any]:
        token = await self._acquire_token()
        url = url_or_path if url_or_path.startswith("http") else f"{GRAPH_BASE}{_ensure_leading_slash(url_or_path)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"Graph GET failed {resp.status}: {body}")
                return body

    async def get_text(self, url_or_path: str) -> str:
        token = await self._acquire_token()
        url = url_or_path if url_or_path.startswith("http") else f"{GRAPH_BASE}{_ensure_leading_slash(url_or_path)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Authorization": f"Bearer {token}"}) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"Graph GET text failed {resp.status}: {body}")
                return body

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = await self._acquire_token()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{GRAPH_BASE}{_ensure_leading_slash(path)}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
            ) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"Graph POST failed {resp.status}: {body}")
                return body

    async def delete(self, path: str) -> None:
        token = await self._acquire_token()
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{GRAPH_BASE}{_ensure_leading_slash(path)}",
                headers={"Authorization": f"Bearer {token}"},
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"Graph DELETE failed {resp.status}: {body}")


class BlobTranscriptStore:
    def __init__(self) -> None:
        container_name = os.environ.get("TRANSCRIPT_BLOB_CONTAINER", "call-transcripts")
        self.container_name = container_name
        service = _build_blob_service_client()
        self.container = service.get_container_client(container_name)
        try:
            self.container.create_container()
        except Exception:
            pass

    def _base_path(self, metadata: dict[str, Any], source_kind: str) -> str:
        tenant_id = (
            metadata.get("meetingOrganizer", {})
            .get("user", {})
            .get("tenantId")
            or os.environ.get("GRAPH_TENANT_ID", "unknown-tenant")
        )
        created = metadata.get("createdDateTime", "")
        date_part = created[:10] if created else datetime.now(UTC).date().isoformat()
        transcript_id = metadata.get("id", "unknown-transcript")
        return f"tenant/{tenant_id}/date/{date_part}/source/{source_kind}/transcript/{transcript_id}"

    async def save_transcript(
        self,
        metadata: dict[str, Any],
        transcript_vtt: str,
        transcript_text: str,
        source_kind: str,
        notification: dict[str, Any],
    ) -> dict[str, str]:
        base_path = self._base_path(metadata, source_kind)
        metadata_blob = f"{base_path}/metadata.json"
        vtt_blob = f"{base_path}/transcript.vtt"
        text_blob = f"{base_path}/transcript.txt"

        enriched_metadata = {
            **metadata,
            "sourceKind": source_kind,
            "savedAt": datetime.now(UTC).isoformat(),
            "transcriptTextBlob": text_blob,
            "transcriptVttBlob": vtt_blob,
            "notificationResource": notification.get("resource"),
            "notificationData": notification,
        }

        self.container.upload_blob(metadata_blob, json.dumps(enriched_metadata, indent=2), overwrite=True)
        self.container.upload_blob(vtt_blob, transcript_vtt, overwrite=True)
        self.container.upload_blob(text_blob, transcript_text, overwrite=True)

        return {"metadata": metadata_blob, "vtt": vtt_blob, "text": text_blob}


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_blob_service_client() -> BlobServiceClient:
    account_name = os.environ.get("TRANSCRIPT_BLOB_ACCOUNT_NAME", "").strip()
    account_key = os.environ.get("TRANSCRIPT_BLOB_ACCOUNT_KEY", "").strip()
    if account_name and account_key:
        account_url = f"https://{account_name}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=account_key)

    connection_string = os.environ.get("TRANSCRIPT_BLOB_CONNECTION_STRING", "").strip()
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    raise RuntimeError(
        "Missing blob credentials. Set TRANSCRIPT_BLOB_CONNECTION_STRING or "
        "TRANSCRIPT_BLOB_ACCOUNT_NAME + TRANSCRIPT_BLOB_ACCOUNT_KEY."
    )


def _ensure_leading_slash(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def _vtt_to_plain_text(content: str) -> str:
    lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"NOTE.*", "", line)
        line = re.sub(r"<v\s+([^>]+)>", r"\1: ", line)
        line = re.sub(r"</?[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _source_kind_from_resource(resource: str) -> str:
    return "adhocCalls" if "adhocCalls" in resource else "onlineMeetings"


def _metadata_content_url(resource: str) -> str:
    resource = resource.split("?")[0].lstrip("/")
    if resource.endswith("/content"):
        return resource
    return f"{resource}/content"


async def _ingest_notification(notification: dict[str, Any], graph: GraphClient, store: BlobTranscriptStore) -> dict[str, Any]:
    resource = notification.get("resource")
    if not resource:
        raise RuntimeError(f"Notification missing resource: {notification}")

    metadata = await graph.get_json(resource)
    transcript_url = metadata.get("transcriptContentUrl")
    if transcript_url:
        transcript_vtt = await graph.get_text(transcript_url)
    else:
        transcript_vtt = await graph.get_text(_metadata_content_url(resource))

    transcript_text = _vtt_to_plain_text(transcript_vtt)
    paths = await store.save_transcript(
        metadata=metadata,
        transcript_vtt=transcript_vtt,
        transcript_text=transcript_text,
        source_kind=_source_kind_from_resource(resource),
        notification=notification,
    )
    return {"resource": resource, "transcriptId": metadata.get("id"), "paths": paths}


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "transcript-ingest"})


async def handle_notifications(request: web.Request) -> web.Response:
    validation_token = request.query.get("validationToken")
    if validation_token:
        return web.Response(text=validation_token, content_type="text/plain")

    body = await request.json()
    notifications = body.get("value", [])
    if not notifications:
        return web.json_response({"status": "ignored", "reason": "no notifications"})

    graph: GraphClient = request.app["graph_client"]
    store: BlobTranscriptStore = request.app["blob_store"]

    async def _process_all() -> None:
        for notification in notifications:
            try:
                result = await _ingest_notification(notification, graph, store)
                logger.info("Ingested transcript %s from %s", result["transcriptId"], result["resource"])
            except Exception:
                logger.exception("Failed to ingest notification: %s", notification)

    asyncio.create_task(_process_all())
    return web.json_response({"status": "accepted", "count": len(notifications)})


async def handle_lifecycle(request: web.Request) -> web.Response:
    validation_token = request.query.get("validationToken")
    if validation_token:
        return web.Response(text=validation_token, content_type="text/plain")

    payload = await request.json()
    logger.info("Received Graph lifecycle event: %s", payload)
    return web.json_response({"status": "ok"})


async def handle_ensure_subscriptions(request: web.Request) -> web.Response:
    graph: GraphClient = request.app["graph_client"]
    payload = await request.json() if request.can_read_body else {}

    resources = payload.get("resources") or os.environ.get("GRAPH_TRANSCRIPT_RESOURCES", ",".join(DEFAULT_RESOURCES)).split(",")
    change_type = payload.get("changeType", "created")
    ttl_minutes = int(payload.get("ttlMinutes", 55))
    notification_url = _require_env("GRAPH_TRANSCRIPT_NOTIFICATION_URL")
    lifecycle_url = os.environ.get("GRAPH_TRANSCRIPT_LIFECYCLE_URL", "")

    created: list[dict[str, Any]] = []
    for raw_resource in resources:
        resource = raw_resource.strip()
        if not resource:
            continue
        subscription_payload: dict[str, Any] = {
            "changeType": change_type,
            "notificationUrl": notification_url,
            "resource": resource,
            "expirationDateTime": (datetime.now(UTC) + timedelta(minutes=ttl_minutes)).isoformat().replace("+00:00", "Z"),
            "clientState": os.environ.get("GRAPH_TRANSCRIPT_CLIENT_STATE", "nat-a365-transcripts"),
        }
        if lifecycle_url:
            subscription_payload["lifecycleNotificationUrl"] = lifecycle_url
        created.append(await graph.post_json("/subscriptions", subscription_payload))

    return web.json_response({"created": created})


async def handle_list_subscriptions(request: web.Request) -> web.Response:
    graph: GraphClient = request.app["graph_client"]
    data = await graph.get_json("/subscriptions")
    resources = request.query.getall("resource", [])
    if resources:
        filtered = []
        for item in data.get("value", []):
            resource = item.get("resource", "")
            if any(filter_value in resource for filter_value in resources):
                filtered.append(item)
        data["value"] = filtered
    return web.json_response(data)


async def handle_delete_subscription(request: web.Request) -> web.Response:
    graph: GraphClient = request.app["graph_client"]
    subscription_id = request.match_info["subscription_id"]
    await graph.delete(f"/subscriptions/{subscription_id}")
    return web.json_response({"deleted": subscription_id})


def build_app() -> web.Application:
    app = web.Application()
    app["graph_client"] = GraphClient()
    app["blob_store"] = BlobTranscriptStore()
    app.add_routes(
        [
            web.get("/health", handle_health),
            web.post("/graph/notifications", handle_notifications),
            web.post("/graph/lifecycle", handle_lifecycle),
            web.post("/admin/subscriptions/ensure", handle_ensure_subscriptions),
            web.get("/admin/subscriptions", handle_list_subscriptions),
            web.delete("/admin/subscriptions/{subscription_id}", handle_delete_subscription),
        ]
    )
    return app


if __name__ == "__main__":
    web.run_app(build_app(), host="0.0.0.0", port=PORT)
