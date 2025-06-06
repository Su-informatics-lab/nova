# Drug-Disease Probability Estimation

Formal retrieval of knowledge from large language models for estimating disease probabilities from medication data (`P(disease|drug)`) 
It is designed to be efficient and robust with multi-seed retries strategy, batching, and checkpointing.


## Supported Assessments

- **diabetes**: Type II diabetes
- **breast_cancer**: Breast cancer  
- **hypertension**: Hypertension

## Key Features

- **Multi-seed retry**: Automatically retries failed predictions with different random seeds
- **Robust checkpointing**: Saves progress every 2 batches, can resume from interruptions
- **GPU optimization**: Supports multi-GPU tensor parallelism and quantization
- **Automatic data loading**: Finds and deduplicates drug files automatically

## Quick Start

```bash
# basic usage
python drug_disease_prob.py --model_name meta-llama/Llama-3.1-8B-Instruct --assessment diabetes

# with reasoning
python drug_disease_prob.py --model_name meta-llama/Llama-3.1-8B-Instruct --assessment breast_cancer --cot

# debug mode (first 200 drugs only)
python drug_disease_prob.py --model_name meta-llama/Llama-3.1-8B-Instruct --assessment diabetes --debug
```

## Requirements

```bash
python3.10 -m venv. venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Data Setup

Place drug data files in the `resources/` folder:
- Files must start with `drugs_` and end with `.parquet`
- Supported columns: `standard_concept_name`, `drug_name`, `drug`, `concept_name`
- Multiple files are automatically merged and deduplicated

## How It Works

1. **Round 1**: Process all drugs with base seed (default: 42)
2. **Round 2+**: Retry failed drugs with offset seeds (1042, 2042, ...)
3. **Continue**: Up to 20 rounds until all drugs succeed or exhaust retries
4. **Checkpoint**: Save progress every 2 batches and after each round

## Arguments

| Argument | Required | Default | Description                                                   |
|----------|----------|---------|---------------------------------------------------------------|
| `--model_name` | ✓ | - | Hugging Face model name                                       |
| `--assessment` | ✓ | - | Assessment type (`diabetes`\|`breast_cancer`\|`hypertension`) |
| `--num_gpus` | | 2 | Number of GPUs for tensor parallelism                         |
| `--temperature` | | 0.6 | Sampling temperature                                          |
| `--seed` | | 42 | Base random seed                                              |
| `--cot` | | False | Enable chain-of-thought reasoning                             |
| `--int4` | | False | Enable 4-bit quantization                                     |
| `--debug` | | False | Process only first 200 drugs                                  |
| `--max_retries` | | 20 | Maximum retry rounds                                          |

## Output

Results are saved as `results/{assessment}_{model}_{options}.parquet` with columns:
- `drug`: Drug name
- `probability`: Estimated probability (0-1, or None if failed)
- `llm_response`: Full LLM response text  
- `seed`: Seed used for this estimate

## Performance Tips

- **Memory issues**: Use `--int4` for large models or reduce `--num_gpus`
- **Speed**: More GPUs = faster processing (if model fits in memory)
- **Reliability**: Default settings provide good balance of speed and success rate

## License

MIT
