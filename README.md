# 🔍 Detecting Illicit Online Promotion with In-Context Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg?logo=huggingface&logoColor=white" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Status-Anonymized_for_Review-orange.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-Academic-green.svg" alt="License">
</p>

<p align="center">
  <b>A Unified Framework for Cross-Platform Illicit Content Detection using In-Context Learning</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Results](#-results)
- [Citation](#-citation)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

Illicit online promotion is a persistent, cross-platform threat that continuously evolves to evade detection. Existing moderation systems remain tethered to platform-specific supervision and static taxonomies—a reactive paradigm that struggles to generalize across domains, adapt to emerging categories, or uncover novel threats before they proliferate.

This repository presents a **systematic study of In-Context Learning (ICL)** as a unified framework for illicit promotion detection across heterogeneous platforms. Through rigorous analysis of prompt design, we establish that properly configured ICL achieves performance comparable to fine-tuned models using **22× fewer labeled examples**.

### 🚀 Key Capabilities

| Capability | Description | Impact |
|:-----------|:------------|:-------|
| **👁️ Seeing the Unseen** | Generalizes to entirely new illicit categories without category-specific demonstrations | < 6% performance drop for 50%+ of 12 evaluated categories |
| **🔬 Autonomous Discovery** | Two-stage pipeline distilling 2,900+ free-form labels into coherent taxonomies | Discovered **8 previously undocumented categories** (usury, illegal immigration, etc.) |
| **🌐 Cross-Platform Generalization** | Deployed on 200K real-world samples without platform adaptation | **92.6% accuracy** with 61.8% uniquely flagged borderline content |

---

## ✨ Key Contributions

1. **📊 Data Efficiency**: ICL matches fine-tuned performance with **22× fewer labeled examples**
2. **🔄 Zero-Shot Generalization**: Maintains robust performance on unseen illicit categories
3. **🆕 Novel Category Discovery**: Uncovered 8 previously undocumented illicit promotion types
4. **🌍 Real-World Deployment**: Validated on 200K samples from search engines and Twitter
5. **⚡ Inference-Time Adaptation**: No retraining required for new platforms or categories

---

## 📁 Project Structure

```
.
├── 📄 ICL_classifier.py              # Main ICL inference script
├── 📄 LoRA_finetuning.py             # LoRA fine-tuning (Unsloth + TRL)
├── 📓 ICL_experiments.ipynb          # Experiment workflows & parameter studies
├── 📓 Visualization.ipynb            # Result analysis & visualization
├── 📂 Data/
│   ├── 📊 balanced_binary_data.csv   # Binary dataset (5,600 samples)
│   └── 📊 balanced_category_data.csv # Multi-class dataset (6,500 samples)
└── 📂 Result/
    ├── 📁 task_binary/
    │   └── results_all.json
    └── 📁 task_multiclass/
        └── results_all.json
```

---

## 📊 Dataset

### Binary Classification Dataset

| Property | Value |
|:---------|:------|
| **File** | `Data/balanced_binary_data.csv` |
| **Samples** | 5,600 |
| **Columns** | `source`, `text`, `label` |
| **Labels** | `benign`, `illicit` |

### Multi-Class Classification Dataset

| Property | Value |
|:---------|:------|
| **File** | `Data/balanced_category_data.csv` |
| **Samples** | 6,500 |
| **Columns** | `source`, `text`, `label`, `language` |
| **Categories** | 12 types including: `porn`, `gambling`, `drug`, `data-theft`, `money-laundry`, `counterfeit`, `advertisement`, `weapon`, `others`, `surrogacy`, `fraud`, `hacking` |

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended for vLLM inference and LoRA fine-tuning)

### Step 1: Create Virtual Environment

```bash
# Using Conda
conda create -n llm-cybercrime python=3.10 -y
conda activate llm-cybercrime

# Or using venv
python3.10 -m venv llm-cybercrime
source llm-cybercrime/bin/activate  # Linux/Mac
# llm-cybercrime\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install torch pandas numpy scikit-learn datasets transformers trl matplotlib

# Additional tools
pip install jieba rank-bm25 retriv vllm psutil

# For optimized fine-tuning
pip install unsloth
```

> ⚠️ **Note**: `unsloth`, `vllm`, and `retriv` can be sensitive to system/CUDA/PyTorch compatibility. If installation fails, please follow their [official installation guides](https://github.com/unslothai/unsloth) for matched versions.

---

## 🚀 Quick Start

### Binary Classification (2-Minute Demo)

```bash
python ICL_classifier.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --train-data ./Data/balanced_binary_data.csv \
  --test-data ./Data/balanced_binary_data.csv \
  --output-path ./Result/icl_binary_predictions.csv \
  --retrieval semantic \
  --n-shots 32 \
  --label-names benign illicit
```

### Expected Output

```
[INFO] Loading model: mistralai/Mistral-7B-Instruct-v0.2
[INFO] Using semantic retrieval with 32 shots
[INFO] Processing 5,600 samples...
[INFO] Accuracy: 0.926 | F1-Score: 0.918
[INFO] Results saved to ./Result/icl_binary_predictions.csv
```

---

## 📖 Usage

### 🔧 ICL Inference (`ICL_classifier.py`)

#### Core Arguments

| Argument | Type | Default | Description |
|:---------|:-----|:--------|:------------|
| `--model-name` | `str` | `mistralai/Mistral-7B-Instruct-v0.2` | Base model name or path |
| `--train-data` | `str` | — | CSV file for demonstration pool |
| `--test-data` | `str` | **Required** | CSV file for testing |
| `--output-path` | `str` | — | Output file path for predictions |
| `--retrieval` | `str` | `semantic` | Retrieval strategy: `random` / `lexical` / `semantic` |
| `--n-shots` | `int` | `32` | Number of demonstrations per class |
| `--label-names` | `list` | — | Space-separated label names |

#### Multi-Class Classification Example

```bash
python ICL_classifier.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --train-data ./Data/balanced_category_data.csv \
  --test-data ./Data/balanced_category_data.csv \
  --output-path ./Result/icl_multiclass_predictions.csv \
  --retrieval semantic \
  --n-shots 32 \
  --label-names porn gambling drug data-theft money-laundry \
              counterfeit advertisement weapon others surrogacy fraud hacking
```

#### Retrieval Strategies

| Strategy | Description | Best For |
|:---------|:------------|:---------|
| `random` | Random demonstration sampling | Baseline comparison |
| `lexical` | BM25 lexical similarity | Keyword-heavy content |
| `semantic` | Embedding-based similarity | **Recommended** - Best overall performance |

### 🎛️ LoRA Fine-Tuning (`LoRA_finetuning.py`)

Fine-tune models with limited labeled data for comparison with ICL:

```bash
python LoRA_finetuning.py -k 1000 -r 42 -e 3
```

| Argument | Short | Description |
|:---------|:------|:------------|
| `--k-shot` | `-k` | Number of training examples to sample |
| `--random-seed` | `-r` | Random seed for reproducibility |
| `--epochs` | `-e` | Number of training epochs |

---

## 📈 Results

### Experiment Summaries

| Task | File Path | Description |
|:-----|:----------|:------------|
| Binary | `Result/task_binary/results_all.json` | Metrics across models, seeds, shots, and retrieval strategies |
| Multi-Class | `Result/task_multiclass/results_all.json` | Comprehensive multi-class evaluation results |

### Performance Highlights

```json
{
  "icl_32shot_semantic": {
    "accuracy": 0.926,
    "f1_score": 0.918,
    "data_efficiency": "22x fewer labels than fine-tuning"
  },
  "zero_shot_generalization": {
    "categories_evaluated": 12,
    "performance_drop": "< 6% for 50%+ categories"
  }
}
```

### 📓 Analysis Notebooks

| Notebook | Purpose |
|:---------|:--------|
| `ICL_experiments.ipynb` | Experiment workflows, parameter studies, ablation analysis |
| `Visualization.ipynb` | Result visualization, performance comparisons, statistical analysis |

---


## ⚠️ Disclaimer

> **🔒 This project is intended for cybersecurity research and academic use only.**

This work focuses on detecting and mitigating illicit online content to improve platform safety. It must **NOT** be used for:
- ❌ Creating or distributing illicit content
- ❌ Circumventing content moderation systems
- ❌ Any illegal activities

By using this software, you agree to use it responsibly and in compliance with all applicable laws and regulations.