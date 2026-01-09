import logging
import typing

async def run_generation(scheduler_address: str,
                         db_url: str,
                         config_file_path: str,
                         job_id: str,
                         payload: typing.Any):
    """Background task to run the workflow."""
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
    from nat.front_ends.fastapi.response_helpers import generate_single_response
    from nat.runtime.loader import load_workflow

    logger = logging.getLogger(__name__)

    job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)
    try:
        async with load_workflow(config_file_path) as local_session_manager:
            async with local_session_manager.session() as session:
                result = await generate_single_response(payload,
                                                        session,
                                                        result_type=session.workflow.single_output_schema)

        del session
        del local_session_manager
        await job_store.update_status(job_id, JobStatus.SUCCESS, output=result)
    except Exception as e:
        logger.exception("Error in async job %s", job_id)
        await job_store.update_status(job_id, JobStatus.FAILURE, error=str(e))
    
    # Explicitly release the resources held by the job store
    del job_store
    