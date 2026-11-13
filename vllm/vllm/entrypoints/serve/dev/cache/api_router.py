# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_running_requests: bool = Query(default=False),
    reset_external: bool = Query(default=False),
):
    """
    Reset the local prefix cache.

    Optionally, if the query parameter `reset_external=true`
    also resets the external (connector-managed) prefix cache.

    Note that we currently do not check if the prefix cache
    is successfully reset in the API server.

    Example:
       POST /reset_prefix_cache?reset_external=true
    """
    logger.info("Resetting prefix cache...")

    await engine_client(raw_request).reset_prefix_cache(
        reset_running_requests, reset_external
    )
    return Response(status_code=200)


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request):
    """
    Reset the multi-modal cache. Note that we currently do not check if the
    multi-modal cache is successfully reset in the API server.
    """
    logger.info("Resetting multi-modal cache...")
    await engine_client(raw_request).reset_mm_cache()
    return Response(status_code=200)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request):
    """
    Reset the encoder cache. Note that we currently do not check if the
    encoder cache is successfully reset in the API server.
    """
    logger.info("Resetting encoder cache...")
    await engine_client(raw_request).reset_encoder_cache()
    return Response(status_code=200)


@router.post("/set_eviction_policy")
async def set_eviction_policy(
    raw_request: Request,
    policy: str = Query(...),
):
    """
    Switch the active GPU KV-cache eviction policy at runtime.

    Only supported when the engine was started with the runtime-switchable
    ``dynamic`` policy (``--gpu-eviction-policy dynamic``), which flips between
    ``lru`` and ``s3_fifo``. Returns whether the active policy changed.

    Example:
       POST /set_eviction_policy?policy=s3_fifo
    """
    logger.info("Setting GPU eviction policy to %s...", policy)
    try:
        changed = await engine_client(raw_request).set_eviction_policy(policy)
    except (ValueError, RuntimeError, NotImplementedError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return JSONResponse(status_code=200, content={"changed": changed})


@router.get("/drain_shadow_trace")
async def drain_shadow_trace(raw_request: Request):
    """
    Return and roll the dynamic-policy shadow trace: the completed-request
    block-hash trace plus the window's average free-block occupancy, for the
    external controller that decides LRU vs. S3-FIFO.

    Only available under ``--gpu-eviction-policy dynamic``.
    """
    try:
        payload = await engine_client(raw_request).drain_shadow_trace()
    except (RuntimeError, NotImplementedError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return JSONResponse(status_code=200, content=payload)


def attach_router(app: FastAPI):
    app.include_router(router)
