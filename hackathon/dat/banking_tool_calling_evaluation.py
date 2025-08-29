#!/usr/bin/env python3
"""
🏦 Banking AI Agent Tool Calling Evaluation

This script evaluates how well a banking AI agent makes tool calls by:
1. Enhancing the banking dataset with realistic tools and expected tool calls
2. Setting up the NeMo Evaluator framework for tool calling evaluation
3. Running the evaluation and analyzing results

Based on the evaluator_tutorial.ipynb structure but adapted for banking scenarios.
"""

import json
import os
import time
from getpass import getpass
from typing import List, Dict, Any

from nemo_microservices import NeMoMicroservices
from huggingface_hub import HfApi


class BankingToolCallingEvaluator:
    """Evaluates banking AI agent tool calling accuracy"""
    
    def __init__(self):
        """Initialize the evaluator with NeMo client and configuration"""
        # Configure microservice host URLs
        self.NEMO_BASE_URL = "https://nmp.aire.nvidia.com"
        self.NIM_BASE_URL = "https://nim.aire.nvidia.com"
        self.DATA_STORE_BASE_URL = "https://datastore.aire.nvidia.com"
        
        # Initialize the client
        self.nemo_client = NeMoMicroservices(
            base_url=self.NEMO_BASE_URL,
            inference_base_url=self.NIM_BASE_URL
        )
        
        # Initialize HF API client
        self.hf_api = HfApi(endpoint=f"{self.DATA_STORE_BASE_URL}/v1/hf", token="")
        
        # Define namespace and dataset names
        self.NAMESPACE = "evaluator-tutorial"  # Use existing working namespace
        self.DATASET_NAME = "banking_tool_calls"
        
    def load_banking_data(self, input_file: str = "banking_agent_evaluation_data.json") -> List[Dict[str, Any]]:
        """Load the banking data that already contains tools and expected tool calls"""
        print("📂 Loading banking data with tools and expected tool calls...")
        
        # Load the data (which should already have tools and tool_calls)
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Verify that the data has the required structure
        for i, record in enumerate(data):
            if "tools" not in record or "tool_calls" not in record:
                print(f"⚠️  Warning: Record {i} missing tools or tool_calls")
                print(f"   Record keys: {list(record.keys())}")
        
        print(f"✅ Loaded {len(data)} records from {input_file}")
        return data
    
    def setup_evaluation_infrastructure(self, enhanced_data: List[Dict[str, Any]]):
        """Set up the evaluation infrastructure in NeMo Data Store"""
        print("🏗️ Setting up evaluation infrastructure...")
        
        # Create dataset repo in datastore
        repo_id = f"{self.NAMESPACE}/{self.DATASET_NAME}"
        try:
            self.hf_api.create_repo(repo_id, repo_type="dataset")
            print(f"✅ Created dataset repo: {repo_id}")
        except Exception as e:
            print(f"ℹ️ Dataset repo already exists or error: {e}")
        
        # Save enhanced data as JSONL for upload
        jsonl_file = "enhanced_banking_tool_calls.jsonl"
        with open(jsonl_file, 'w') as f:
            for record in enhanced_data:
                f.write(json.dumps(record) + '\n')
        
        # Upload the dataset
        try:
            self.hf_api.upload_file(
                repo_type="dataset",
                repo_id=repo_id,
                revision="main",
                path_or_fileobj=jsonl_file,
                path_in_repo="banking_tool_calls.jsonl"
            )
            print(f"✅ Uploaded dataset to Data Store")
        except Exception as e:
            print(f"ℹ️ Dataset upload error (may already exist): {e}")
        
        # Create dataset in NeMo
        try:
            response = self.nemo_client.datasets.create(
                name=self.DATASET_NAME,
                namespace=self.NAMESPACE,
                description="Banking tool calling evaluation dataset",
                files_url=f"hf://datasets/{self.NAMESPACE}/{self.DATASET_NAME}",
                project="banking-evaluation",
                custom_fields={},
            )
            print(f"✅ Created dataset in NeMo: {response.id}")
        except Exception as e:
            print(f"ℹ️ Dataset creation error (may already exist): {e}")
        
        # Clean up temporary file
        if os.path.exists(jsonl_file):
            os.remove(jsonl_file)
    
    def create_evaluation_target(self):
        """Create an evaluation target for the banking AI agent"""
        print("🎯 Getting evaluation target...")
        
        try:
            # Use the existing llama-chat-target that we know works
            targets = self.nemo_client.evaluation.targets.list()
            for target in targets.data:
                if (hasattr(target, 'name') and 
                    target.name == "llama-chat-target" and 
                    hasattr(target, 'namespace') and 
                    target.namespace == "evaluator-tutorial"):
                    print(f"✅ Using existing target: {target.id} ({target.name})")
                    return target
            
            print("❌ Could not find llama-chat-target in evaluator-tutorial namespace")
            return None
            
        except Exception as e:
            print(f"❌ Error getting evaluation target: {e}")
            return None
    
    def create_evaluation_config(self):
        """Create evaluation configuration for tool calling accuracy"""
        print("⚙️ Getting evaluation configuration...")
        
        try:
            # Use the existing tool-call-eval-config that we know works
            configs = self.nemo_client.evaluation.configs.list()
            for config in configs.data:
                if (hasattr(config, 'name') and 
                    config.name == "tool-call-eval-config"):
                    print(f"✅ Using existing config: {config.id} ({config.name})")
                    return config
            
            print("❌ Could not find tool-call-eval-config")
            return None
            
        except Exception as e:
            print(f"❌ Error getting evaluation config: {e}")
            return None
    
    def run_evaluation(self, target, config):
        """Run the tool calling evaluation"""
        print("🚀 Running tool calling evaluation...")
        
        try:
            # Create evaluation job
            job = self.nemo_client.evaluation.jobs.create(
                namespace=self.NAMESPACE,
                target=f"{self.NAMESPACE}/llama-chat-target",
                config=f"{config.namespace}/{config.name}"
            )
            
            job_id = job.id
            print(f"✅ Created evaluation job: {job_id}")
            
            # Wait for completion
            print("⏳ Waiting for evaluation to complete...")
            max_wait_time = 300  # 5 minutes max wait
            start_time = time.time()
            
            while True:
                status = self.nemo_client.evaluation.jobs.status(job_id)
                
                # Debug: Print the entire status object
                print(f"🔍 Raw status object: {status}")
                print(f"🔍 Status object type: {type(status)}")
                print(f"🔍 Status object attributes: {dir(status)}")
                
                # Handle different possible status attribute names
                if hasattr(status, 'status'):
                    status_value = status.status
                    print(f"✅ Found 'status' attribute: {status_value}")
                elif hasattr(status, 'state'):
                    status_value = status.state
                    print(f"✅ Found 'state' attribute: {status_value}")
                else:
                    print(f"⚠️ No status/state attribute found")
                    # Try to get status from the object itself
                    status_value = str(status).lower()
                    if "completed" in status_value:
                        status_value = "completed"
                    elif "failed" in status_value:
                        status_value = "failed"
                    elif "running" in status_value:
                        status_value = "running"
                    else:
                        status_value = "unknown"
                
                # Handle progress attribute
                progress = getattr(status, 'progress', 'unknown')
                print(f"📊 Job status: {status_value} - Progress: {progress}%")
                
                # Check for completion
                if status_value == "completed" or "completed" in str(status).lower():
                    print("🎉 Evaluation completed successfully!")
                    break
                elif status_value == "failed" or "failed" in str(status).lower():
                    print("❌ Evaluation failed!")
                    return None
                elif status_value == "running" or "running" in str(status).lower():
                    print("🔄 Evaluation is running...")
                elif status_value == "pending" or "pending" in str(status).lower():
                    print("⏳ Evaluation is pending...")
                else:
                    print(f"🤔 Unknown status: {status_value}")
                
                # Check if we've been waiting too long
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    print(f"⏰ Timeout after {max_wait_time} seconds. Stopping wait.")
                    break
                
                print(f"⏳ Waiting 30 seconds... (elapsed: {elapsed_time:.0f}s)")
                time.sleep(30)  # Wait 30 seconds before checking again
            
            # Get results
            results = self.nemo_client.evaluation.jobs.results(job_id)
            print("📈 Evaluation Results:")
            print(json.dumps(results.tasks, indent=2, default=str))
            
            return results
            
        except Exception as e:
            print(f"❌ Error running evaluation: {e}")
            return None
    
    def analyze_results(self, results):
        """Analyze and display evaluation results"""
        if not results:
            print("❌ No results to analyze")
            return
        
        print("\n🔍 Detailed Results Analysis:")
        print("=" * 50)
        
        # Prepare report data
        report_data = {
            "evaluation_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_type": "Banking AI Agent Tool Calling",
                "total_scenarios": 0,
                "overall_score": 0.0
            },
            "detailed_metrics": {},
            "interpretation": {},
            "recommendations": []
        }
        
        # Extract metrics - handle different possible task names
        task_name = None
        for possible_name in ['banking-tool-calling', 'custom-tool-calling', 'tool-calling']:
            if possible_name in results.tasks:
                task_name = possible_name
                break
        
        if task_name:
            task_result = results.tasks[task_name]
            print(f"📊 Processing task: {task_name}")
            
            if 'tool-calling-accuracy' in task_result.metrics:
                metric = task_result.metrics['tool-calling-accuracy']
                
                print(f"📊 Tool Calling Accuracy Metrics:")
                report_data["detailed_metrics"]["tool_calling_accuracy"] = {}
                
                total_score = 0
                metric_count = 0
                
                for score_name, score_result in metric.scores.items():
                    score_value = score_result.value
                    total_score += score_value
                    metric_count += 1
                    
                    print(f"  {score_name}: {score_value}")
                    report_data["detailed_metrics"]["tool_calling_accuracy"][score_name] = {
                        "score": score_value,
                        "stats": {}
                    }
                    
                    if hasattr(score_result, 'stats') and score_result.stats:
                        stats = score_result.stats
                        if hasattr(stats, 'count') and stats.count:
                            print(f"    Sample count: {stats.count}")
                            report_data["detailed_metrics"]["tool_calling_accuracy"][score_name]["stats"]["count"] = stats.count
                        if hasattr(stats, 'mean') and stats.mean:
                            print(f"    Mean: {stats.mean}")
                            report_data["detailed_metrics"]["tool_calling_accuracy"][score_name]["stats"]["mean"] = stats.mean
                        if hasattr(stats, 'stddev') and stats.stddev:
                            print(f"    Standard deviation: {stats.stddev}")
                            report_data["detailed_metrics"]["tool_calling_accuracy"][score_name]["stats"]["stddev"] = stats.stddev
                
                # Calculate overall score
                if metric_count > 0:
                    overall_score = total_score / metric_count
                    report_data["evaluation_summary"]["overall_score"] = overall_score
                    report_data["evaluation_summary"]["total_scenarios"] = metric_count
        
        # Add interpretation
        print("\n🎯 Interpretation:")
        report_data["interpretation"] = {
            "score_ranges": {
                "perfect": "1.0 - AI agent makes exactly the right tool calls",
                "high": "0.8-0.99 - AI agent makes mostly correct tool calls", 
                "medium": "0.6-0.79 - AI agent makes some correct tool calls",
                "low": "<0.6 - AI agent struggles with tool calling"
            }
        }
        
        print("- Perfect score (1.0): AI agent makes exactly the right tool calls")
        print("- High score (0.8-0.99): AI agent makes mostly correct tool calls")
        print("- Medium score (0.6-0.79): AI agent makes some correct tool calls")
        print("- Low score (<0.6): AI agent struggles with tool calling")
        
        # Generate recommendations based on results
        if 'tool_calling_accuracy' in report_data["detailed_metrics"]:
            recommendations = []
            
            if report_data["detailed_metrics"]["tool_calling_accuracy"].get("function_name_accuracy", {}).get("score", 0) < 0.8:
                recommendations.append("Improve function name selection accuracy")
            
            if report_data["detailed_metrics"]["tool_calling_accuracy"].get("function_name_and_args_accuracy", {}).get("score", 0) < 0.8:
                recommendations.append("Enhance parameter extraction and formatting")
            
            if not recommendations:
                recommendations.append("Excellent performance! Consider testing with more diverse scenarios")
            
            report_data["recommendations"] = recommendations
            
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        # Save report to file
        self.save_evaluation_report(report_data, results)
        
        return report_data
    
    def save_evaluation_report(self, report_data, results):
        """Save the evaluation report to a file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"banking_evaluation_report_{timestamp}.json"
        
        # Create comprehensive report
        full_report = {
            "report_metadata": {
                "generated_at": report_data["evaluation_summary"]["timestamp"],
                "script_version": "1.0",
                "evaluation_type": "Banking AI Agent Tool Calling"
            },
            "evaluation_summary": report_data["evaluation_summary"],
            "detailed_metrics": report_data["detailed_metrics"],
            "interpretation": report_data["interpretation"],
            "recommendations": report_data["recommendations"],
            "raw_results": str(results.tasks) if hasattr(results, 'tasks') else str(results)
        }
        
        # Save JSON report
        with open(report_filename, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        print(f"\n📄 Evaluation report saved to: {report_filename}")
        
        # Also create a human-readable text report
        text_report_filename = f"banking_evaluation_report_{timestamp}.txt"
        with open(text_report_filename, 'w') as f:
            f.write("🏦 BANKING AI AGENT TOOL CALLING EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📅 Generated: {report_data['evaluation_summary']['timestamp']}\n")
            f.write(f"🎯 Evaluation Type: {report_data['evaluation_summary']['evaluation_type']}\n")
            f.write(f"📊 Total Scenarios: {report_data['evaluation_summary']['total_scenarios']}\n")
            f.write(f"⭐ Overall Score: {report_data['evaluation_summary']['overall_score']:.2f}\n\n")
            
            f.write("📈 DETAILED METRICS\n")
            f.write("-" * 30 + "\n")
            if 'tool_calling_accuracy' in report_data['detailed_metrics']:
                for metric_name, metric_data in report_data['detailed_metrics']['tool_calling_accuracy'].items():
                    f.write(f"  {metric_name}: {metric_data['score']:.2f}\n")
                    if 'stats' in metric_data and metric_data['stats']:
                        for stat_name, stat_value in metric_data['stats'].items():
                            f.write(f"    {stat_name}: {stat_value}\n")
            f.write("\n")
            
            f.write("🎯 INTERPRETATION\n")
            f.write("-" * 20 + "\n")
            for score_range, description in report_data['interpretation']['score_ranges'].items():
                f.write(f"  {score_range.title()}: {description}\n")
            f.write("\n")
            
            f.write("💡 RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for rec in report_data['recommendations']:
                f.write(f"  - {rec}\n")
            f.write("\n")
            
            f.write("📋 NEXT STEPS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Review the detailed results above\n")
            f.write("2. Identify areas where your banking AI agent needs improvement\n")
            f.write("3. Iterate on your agent's tool calling logic\n")
            f.write("4. Re-run evaluation to measure improvements\n")
            f.write("5. Scale up testing with more diverse banking scenarios\n")
        
        print(f"📄 Human-readable report saved to: {text_report_filename}")
        
        return report_filename, text_report_filename
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("🏦 Banking AI Agent Tool Calling Evaluation")
        print("=" * 60)
        
        # Step 1: Load the banking data
        enhanced_data = self.load_banking_data()
        
        # Step 2: Set up infrastructure
        self.setup_evaluation_infrastructure(enhanced_data)
        
        # Step 3: Create evaluation target
        target = self.create_evaluation_target()
        if not target:
            print("❌ Failed to create/get evaluation target")
            return
        
        # Step 4: Create evaluation config
        config = self.create_evaluation_config()
        if not config:
            print("❌ Failed to create/get evaluation config")
            return
        
        # Step 5: Run evaluation
        results = self.run_evaluation(target, config)
        
        # Step 6: Analyze results
        self.analyze_results(results)
        
        print("\n🎉 Evaluation pipeline completed!")
        print("\n📋 Next Steps:")
        print("1. Review the detailed results above")
        print("2. Identify areas where your banking AI agent needs improvement")
        print("3. Iterate on your agent's tool calling logic")
        print("4. Re-run evaluation to measure improvements")


def main():
    """Main function to run the banking tool calling evaluation"""
    evaluator = BankingToolCallingEvaluator()
    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()
