# Launchables for NeMo-Agent-Toolkit

Launchables are an easy way to bundle a hardware and software environment into an easily shareable link, allowing for simple demos of GPU-powered software. 

NeMo-Agent-Toolkit offers the following notebooks in Launchable format: 

1. [GPU Cluster Sizing with NeMo-Agent-Toolkit](https://github.com/nv-edwli/NeMo-Agent-Toolkit/blob/develop/examples/notebooks/launchables/GPU_Cluster_Sizing_with_NeMo_Agent_Toolkit.ipynb) [![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-31vuPuKGq9uMQg4SkwBnSJYFOqN)

    * **TODO:** Update the Deploy Badge to the correct/final Launchable link
    * This notebook demonstrates how to use the NVIDIA NeMo Agent toolkit's sizing calculator to estimate the GPU cluster size required to accommodate a target number of users with a target response time. The estimation is based on the performance of the workflow at different concurrency levels.
    * The sizing calculator uses the [evaluation](https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/evaluate.html) and [profiling](https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/profiler.html) systems in the NeMo Agent toolkit.
