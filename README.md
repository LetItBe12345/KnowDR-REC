# KnowDR-REC Evaluation Framework

[![paper](https://img.shields.io/badge/arXiv-Paper-blue.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Available-F9D371)](https://drive.google.com/drive/folders/1QiRKtHgkZ8sBaeZF_g9Sq7cRdfUrRipM?usp=sharing)
[![Code](https://img.shields.io/badge/Code-Open_Source-green.svg)](https://github.com/LetItBe12345/KnowDR-REC)

This repository contains the evaluation framework for **KnowDR-REC**, a novel benchmark for knowledge-driven referring expression comprehension that challenges large multimodal models with real-world knowledge reasoning and fine-grained visual grounding.

> **Abstract:** *Referring Expression Comprehension (REC) is a popular multimodal task that aims to accurately detect target objects within a single image based on a given textual expression. However, due to the limitations of earlier models, traditional REC benchmarks either rely solely on intra-image cues or lack sufficiently fine-grained instance annotations, making them inadequate for evaluating the reasoning capabilities of large multimodal models (LMMs). To address this gap, we propose a new benchmark, KnowDR-REC, characterized by three key features: Firstly, it is built upon real-world knowledge, requiring fine-grained multimodal reasoning across text and image. Secondly, the dataset includes elaborately constructed negative samples via fine-grained expression editing, designed to evaluate a model's robustness and anti-hallucination ability. Lastly, we introduce three novel evaluation metrics to systematically explore the model's internal reasoning process. We evaluate 16 state-of-the-art multimodal models on KnowDR-REC, with experimental results showing that existing LMMs still struggle with knowledge-driven visual grounding tasks. Furthermore, we observe a decoupling between textual understanding and visual grounding in LMMs, where many models are significantly influenced by memorized shortcut correlations, which severely affect their behavior on our benchmark and hinder genuine multimodal reasoning. We anticipate that the proposed benchmark will inspire future research towards developing more robust, interpretable, and knowledge-intensive visual grounding frameworks, driving the development of more reliable and robust multimodal systems for complex real-world scenarios.*

## Key Features

🧠 **Knowledge-Driven**: Built upon real-world knowledge requiring fine-grained multimodal reasoning  
📊 **Novel Metrics**: Three new evaluation metrics to explore internal reasoning processes  


## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/KnowDR-REC.git
cd KnowDR-REC/evaluation

# Install dependencies
pip install aiohttp pyyaml pillow numpy tqdm

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

## Quick Start

### 1. Configure Your Evaluation

Edit `config/prompts.yaml` to define your prompt templates.


### 2. Run Evaluation

```bash
# Basic evaluation with all 4 pre-configured models
python run_evaluation.py --dataset /path/to/your/dataset.json

# Evaluate specific models
python run_evaluation.py \
  --dataset /path/to/dataset.json \
  --models gemini-2.5-flash qwen-2.5-vl-7b \
  --output my_results \
  --max-samples 100
```

### 3. View Results

```bash
# Results are saved in structured directories
evaluation_results/
├── model_gemini-2.5-flash/
│   ├── evaluation_openrouter_gemini-2.5-flash_*.json
│   └── evaluation.log
├── model_qwen-2.5-vl-7b/
│   └── ...
└── evaluation.log
```

## Supported Models

The framework comes pre-configured with 4 representative vision language models:

| Model | Provider | Description | Performance Tier |
|-------|----------|-------------|------------------|
| `gemini-2.5-flash` | Google | Fast response, cost-effective | High Speed |
| `gemini-2.5-pro` | Google | Professional grade, highest quality | High Performance |
| `grok-2-vision` | X.AI | Innovative vision-language model | Cutting Edge |
| `qwen-2.5-vl` | Alibaba | Open-source, high value | Cost Effective |

### Adding Custom Models

Simply add any OpenRouter-supported model to `config/models.yaml`:

```yaml
user_models:
  - name: "claude-3.5-sonnet"
    openrouter_id: "anthropic/claude-3.5-sonnet"
    max_tokens: 2048
    temperature: 0.1
    recommended_concurrent: 3
```

## Dataset Format

KnowDR-REC uses a JSON format compatible with standard referring expression datasets:

```json
[
  {
    "image_path": "./folder_directory/file1_images/image1_01.jpg",
    "type": "negative",
    "source": "complexwebquestions",
    "text": "The person was the 2019 Ravens Quarterback who completed the largest passes.",
    "licenses": "ComplexWebQuestions (Talmor and Berant 2018) Apache-2.0"
  },
  {
    "image_path": "./folder_directory/file1_images/image1_02.jpg",
    "type": "negative",
    "source": "hotpotqa",
    "text": "The person is the older brother of the star of the 1997 film \"The Good Life\".",
    "licenses": "HotpotQA (Yang et al. 2018) CC BY-SA 4.0"
  }
]
```


## Configuration

### Runtime Configuration (`config/config_template.yaml`)

```yaml
api:
  openrouter_api_key: "${OPENROUTER_API_KEY}"

models:
  - "gemini-2.5-flash"
  - "gemini-2.5-pro" 
  - "grok-2-vision"
  - "qwen-2.5-vl-7b"

evaluation:
  max_samples: null          # null = evaluate all samples
  batch_size: 32
  concurrent_requests: 5
  output_dir: "evaluation_results"
```

### Advanced Usage

```bash
# Custom configuration file
cp config/config_template.yaml my_config.yaml
# Edit my_config.yaml as needed
python run_evaluation.py --config my_config.yaml --dataset dataset.json

# Override specific parameters
python run_evaluation.py \
  --config my_config.yaml \
  --dataset dataset.json \
  --models gemini-2.5-pro \
  --max-samples 50 \
  --output quick_test \
  --verbose
```



## Contact

For questions about the benchmark, evaluation framework, or research collaboration:

- 🐛 Issues: [GitHub Issues](https://github.com/LetItBe12345/KnowDR-REC/issues)


