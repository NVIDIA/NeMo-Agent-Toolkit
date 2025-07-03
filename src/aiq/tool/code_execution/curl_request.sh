#!/bin/bash

# Curl request to send the generated code JSON payload
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @- \
  "http://127.0.0.1:6000/execute" << 'EOF'
{
  "generated_code": "\nimport traceback\nimport json\nimport os\nimport warnings\nimport contextlib\nimport io\nwarnings.filterwarnings(\'ignore\')\nos.environ[\'OPENBLAS_NUM_THREADS\'] = \'16\'\n\n\ngenerated_code = \'{\\\'generated_code\\\': \\\'### START PYTHON CODE ###\\\\nimport pandas as pd\\\\nimport plotly.graph_objects as go\\\\n\\\\ndata = pd.read_json(\\\\\\\'retrieve_time_in_cycles_and_se_results.json\\\\\\\')\\\\n\\\\nfig = go.Figure(data=[go.Scatter(x=data[\\\\\\\'time_in_cycles\\\\\\\'], y=data[\\\\\\\'sensor_measurement_13\\\\\\\'], mode=\\\\\\\'lines+markers\\\\\\\')])\\\\nfig.update_layout(title=\\\\\\\'Sensor Measurement 13 vs Time for Engine 4\\\\\\\', xaxis_title=\\\\\\\'Time in Cycles\\\\\\\', yaxis_title=\\\\\\\'Sensor Measurement 13\\\\\\\')\\\\n\\\\nfig.write_html(\\\\\\\'engine_4_sensor_13_vs_time.html\\\\\\\')\\\\nprint(f"Plot saved to: engine_4_sensor_13_vs_time.html")\\\\n### END PYTHON CODE ###\\\'}\'\n\nstdout = io.StringIO()\nstderr = io.StringIO()\n\nwith contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):\n    try:\n        exec(generated_code)\n        status = "completed"\n    except Exception:\n        status = "error"\n        stderr.write(traceback.format_exc())\nstdout = stdout.getvalue()\nstderr = stderr.getvalue()\nif len(stdout) > 2000:\n    stdout = stdout[:2000] + "<output cut>"\nif len(stderr) > 2000:\n    stderr = stderr[:2000] + "<output cut>"\nif stdout:\n    stdout += "\\n"\nif stderr:\n    stderr += "\\n"\noutput = {"process_status": status, "stdout": stdout, "stderr": stderr}\nprint(json.dumps(output))\n", 
  "timeout": 10.0, 
  "language": "python"
}
EOF




curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @- \
  "http://127.0.0.1:6000/execute" << 'EOF'
{
  "generated_code": "### START PYTHON CODE ###\nprint('Hello, World!')### END PYTHON CODE ###",
  "timeout" : 100
}
EOF

# # Load data using relative path (working directory is /workspace mounted to your output_data)
# data = pd.read_json('retrieve_time_in_cycles_and_se_results.json')

# # Create your analysis/plot
# fig = go.Figure(data=[go.Scatter(x=data['time_in_cycles'], 
#                                 y=data['sensor_measurement_10'], 
#                                 mode='lines+markers')])

# fig.update_layout(
#     title='Sensor Measurement 10 vs Time for Engine 4',
#     xaxis_title='Time in Cycles',
#     yaxis_title='Sensor Measurement 10'
# )

# # Save to current directory (will appear in your local output_data folder)
# fig.write_html('engine_4_sensor_10_vs_time.html')
# print(f\"Plot saved to: engine_4_sensor_10_vs_time.html\")