import os
import json
import logging
from pathlib import Path
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torchaudio
from torchvision import transforms

from typing import List, Dict, Callable, Any

# =======================
# LOGGING
# =======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =======================
# TOOL SYSTEM
# =======================
class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description

    def execute(self, *args, **kwargs):
        logging.info(f"[Tool] Running {self.name}")
        return self.func(*args, **kwargs)


# =======================
# MEMORY SYSTEM
# =======================
class Memory:
    def __init__(self):
        self.storage: List[Dict] = []

    def add(self, content: str):
        self.storage.append({"content": content})

    def search(self, query: str):
        return [
            item["content"]
            for item in self.storage
            if query.lower() in item["content"].lower()
        ]


# =======================
# SIMPLE AI MODEL (PYTORCH)
# =======================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =======================
# RESEARCH AGENT
# =======================
class ResearchAgent:
    def __init__(self, name: str):
        self.name = name
        self.memory = Memory()
        self.tools: Dict[str, Tool] = {}
        self.goals: List[str] = []

        # AI Model
        self.model = SimpleNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    # ---------- TOOLS ----------
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def use_tool(self, tool_name: str, *args):
        if tool_name not in self.tools:
            raise ValueError("Tool not found")

        result = self.tools[tool_name].execute(*args)
        self.memory.add(str(result))
        return result

    # ---------- GOALS ----------
    def add_goal(self, goal: str):
        self.goals.append(goal)

    # ---------- DATA ANALYSIS ----------
    def analyze_csv(self, file_path: str):
        df = pd.read_csv(file_path)

        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "mean": df.mean(numeric_only=True).to_dict()
        }

        logging.info(f"[Agent] Data analyzed: {summary}")
        self.memory.add(json.dumps(summary))
        return summary

    # ---------- VISUALIZATION ----------
    def visualize_data(self, data: List[int]):
        plt.figure()
        plt.plot(data)
        plt.title("Data Visualization")
        plt.show()

    # ---------- TRAIN MODEL ----------
    def train_model(self):
        # Dummy data
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

        for epoch in range(100):
            pred = self.model(x)
            loss = F.mse_loss(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logging.info(f"[AI] Training complete. Loss: {loss.item()}")
        return loss.item()

    # ---------- AUDIO PROCESSING ----------
    def process_audio(self, audio_path: str):
        waveform, sample_rate = torchaudio.load(audio_path)
        return {
            "shape": waveform.shape,
            "sample_rate": sample_rate
        }

    # ---------- FILE HANDLING ----------
    def unzip_file(self, zip_path: str, extract_to: str):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return f"Extracted to {extract_to}"

    # ---------- THINK ----------
    def think(self, query: str):
        logging.info(f"[Thinking] {query}")

        memory = self.memory.search(query)
        if memory:
            return {"memory": memory}

        research = f"Researching {query}..."
        opinion = f"{query} is a growing trend in AI."

        self.memory.add(research)
        self.memory.add(opinion)

        return {
            "research": research,
            "opinion": opinion
        }


# =======================
# SAMPLE TOOL FUNCTIONS
# =======================
def numpy_analysis(data: List[int]):
    arr = np.array(data)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr)
    }


def file_exists(path: str):
    return os.path.exists(path)


# =======================
# MAIN
# =======================
if __name__ == "__main__":
    agent = ResearchAgent("Dr. AI")

    # Add tools
    agent.add_tool(Tool("NumpyAnalyzer", numpy_analysis, "Analyze using numpy"))
    agent.add_tool(Tool("FileChecker", file_exists, "Check file existence"))

    # Goals
    agent.add_goal("Master AI + Data + Automation")

    print("\n--- TOOL ---")
    print(agent.use_tool("NumpyAnalyzer", [10, 20, 30]))

    print("\n--- TRAIN AI ---")
    agent.train_model()

    print("\n--- THINK ---")
    print(agent.think("Future of AI"))

    print("\n--- FILE CHECK ---")
    print(agent.use_tool("FileChecker", "test.csv"))