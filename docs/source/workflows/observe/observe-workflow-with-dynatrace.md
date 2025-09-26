# Observing a Workflow with Dynatrace

This guide shows how to stream OpenTelemetry (OTel) traces from your NVIDIA NeMo Agent toolkit workflows to the [OpenTelemetry Protocol (OTLP) ingest API](https://docs.dynatrace.com/docs/discover-dynatrace/references/dynatrace-api/environment-api/opentelemetry), which in turn provides the ability to have full visibility into the performance of AI/LLM models and agent interactions​. 

In this guide, you will learn how to:
* Deploy a [Dynatrace OpenTelemetry Collector](https://docs.dynatrace.com/docs/ingest-from/opentelemetry/collector) with a configuration that exports traces into Dynatrace
* Configure your workflow (YAML) or Python script to send traces to the OTel collector.
* Run the workflow and view traces within Dynatrace

## Step 1: Dynatrace Account
View the exported traces within the [Dynatrace Distributed Tracing App](https://docs.dynatrace.com/docs/analyze-explore-automate/distributed-tracing/distributed-tracing-app) as shown below.


<div align="left">
  <img src="../../_static/dynatrace-trace.png" width="800">
</div>