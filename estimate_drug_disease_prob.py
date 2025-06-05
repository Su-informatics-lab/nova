"""
This script estimates the probability of various medical conditions based on 
medication data using LLMs with a multi-seed retry approach.

Supported Assessments:
- diabetes: Type II diabetes
- breast_cancer: Breast cancer  
- hypertension: Hypertension

Multi-Seed Retry Strategy:
- Processes all drugs with base seed, collects failures
- Retries failed drugs with offset seeds (base_seed + round*1000)  
- Continues for up to max_retries rounds until all drugs succeed or exhaust retries
- Saves progress after each round for robust checkpointing

Usage:
    python estimate_drug_disease_prob.py --model_name MODEL_NAME \
    --assessment ASSESSMENT_TYPE [options]

Required Arguments:
    --model_name     Huggingface model name (e.g., meta-llama/Llama-3.1-70B-Instruct)
    --assessment     Assessment type (diabetes|breast_cancer|hypertension)

Optional Arguments:
    --cot           Enable chain-of-thought reasoning
    --num_gpus      Number of GPUs for tensor parallelism (default: 2)
    --temperature   Sampling temperature (default: 0.6)
    --seed          Base random seed for reproducibility (default: 42)
    --debug         Enable debug mode (process only first 200 drugs and print responses)
    --int4          Enable BitsAndBytes 4-bit quantization
    --max_retries   Maximum number of retry rounds with different seeds (default: 20)

Input:
    Automatically loads and deduplicates all parquet files in 'resources/' folder
    starting with "drugs_"

Output:
    Generates a parquet file containing:
    - drug: Drug name
    - probability: Estimated probability (0-1, or None if permanently failed)
    - llm_response: Full LLM response text
    - seed: Seed used for this particular estimate
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch import manual_seed
from tqdm import tqdm
from vllm import LLM, SamplingParams

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
MAX_MODEL_LENGTH = 4096
MODEL_NAME_GLOBAL = None


@dataclass
class AssessmentConfig:
    name: str
    system_prompt: str
    question: str

    def create_prompt(self, drug: str, cot: bool = False) -> str:
        """
        Create a simple, direct prompt for the assessment.
        Args:
            drug: Name of the drug
            cot: Whether to include chain-of-thought reasoning instruction
        """
        questionnaire_info = f"\n\n{self.question}\n\n" if self.question else "\n\n"
        base_prompt = (
            f"Given that a patient took {drug}, estimate the probability that they have {self.name}."
            f"{questionnaire_info}"
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )

        if cot:
            base_prompt += "\nYou may think aloud and reason step-by-step before reaching the final answer."

        return base_prompt


ASSESSMENT_CONFIGS = {
    "diabetes": AssessmentConfig(
        name="Type II diabetes",
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has Type II diabetes based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "breast_cancer": AssessmentConfig(
        name="breast cancer",
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a woman has breast cancer based solely on medication data. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "hypertension": AssessmentConfig(
        name="hypertension",
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has hypertension based solely on medication data. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
}


def create_conversation(
    drug: str, assessment_config: AssessmentConfig, cot: bool
) -> List[Dict]:
    """
    Create a conversation template.
    For models that do not support a separate system role (e.g., DeepSeek-R1 and Gemma-2),
    prepend the system instruction to the user prompt.
    """
    system_prompt = assessment_config.system_prompt
    # add extra directive for chain-of-thought if using deepseek-r1 model
    # refer to https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    if "deepseek-r1" in MODEL_NAME_GLOBAL:
        system_prompt += '\nPlease ensure that your answer begins with "<think>\n".'
    user_prompt = assessment_config.create_prompt(drug, cot)
    # combine system and user prompts for models that don't support system role
    if ("deepseek-r1" in MODEL_NAME_GLOBAL) or ("gemma" in MODEL_NAME_GLOBAL):
        combined_prompt = f"{system_prompt}\n{user_prompt}"
        return [{"role": "user", "content": combined_prompt}]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


def extract_probability(response_text: str) -> Optional[float]:
    """Extract probability from LLM response that uses [ESTIMATION] tags."""
    if not response_text:
        return None
    tag_match = re.search(
        r"\[ESTIMATION\](.*?)\[/ESTIMATION\]", response_text, re.DOTALL
    )
    if not tag_match:
        return None
    estimation_text = tag_match.group(1).strip()
    try:
        value = float(estimation_text)
        if not np.isfinite(value):
            return None
        if 0 <= value <= 1:
            return value
        return None
    except ValueError:
        # try to extract percentage format like "25%"
        percentage_match = re.search(r"(\d+(?:\.\d+)?)%", estimation_text)
        if percentage_match:
            try:
                value = float(percentage_match.group(1)) / 100
                if np.isfinite(value) and 0 <= value <= 1:
                    return value
            except ValueError:
                pass
    return None


def generate_batch_conversations(
    drugs: List[str], assessment_config: AssessmentConfig, cot: bool
) -> List[List[Dict]]:
    """Generate batch conversations for vLLM processing."""
    conversations = []
    for drug in drugs:
        conversation = create_conversation(drug, assessment_config, cot)
        conversations.append(conversation)
    return conversations


def process_batch_results(
    outputs: List, drugs: List[str], current_seed: int
) -> tuple[List[Dict], Set[str]]:
    """
    Process batch results from vLLM.
    Returns:
        - List of successful results
        - Set of failed drugs that need retry
    """
    results = []
    failed_drugs = set()

    for drug, output in zip(drugs, outputs):
        try:
            if output and output.outputs and len(output.outputs) > 0:
                response_text = output.outputs[0].text
                probability = extract_probability(response_text)

                if probability is not None:
                    # successfully extracted probability
                    result = {
                        "drug": drug,
                        "probability": probability,
                        "llm_response": response_text,
                        "seed": current_seed,
                    }
                    results.append(result)
                else:
                    # failed to extract probability
                    logging.warning(
                        f"Failed to extract probability for {drug} (seed: {current_seed})"
                    )
                    failed_drugs.add(drug)
            else:
                # empty output
                logging.warning(f"Empty output for {drug} (seed: {current_seed})")
                failed_drugs.add(drug)
        except Exception as e:
            logging.error(
                f"Error processing result for {drug} (seed: {current_seed}): {str(e)}"
            )
            failed_drugs.add(drug)

    return results, failed_drugs


def save_checkpoint(
    results_df: pd.DataFrame, new_results: List[Dict], checkpoint_file: str
) -> pd.DataFrame:
    """Save checkpoint and return updated dataframe."""
    try:
        if new_results:
            updated_df = pd.concat(
                [results_df, pd.DataFrame(new_results)], ignore_index=True
            )
        else:
            updated_df = results_df
        updated_df.to_parquet(checkpoint_file)
        return updated_df
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        return results_df


def load_drug_data() -> List[str]:
    """Load and deduplicate drug data from all parquet files starting with 'drugs_' in resources folder."""
    resources_dir = "resources"

    if not os.path.exists(resources_dir):
        raise FileNotFoundError(f"Resources directory '{resources_dir}' not found")

    # find all parquet files starting with "drugs_"
    drug_files = [
        f
        for f in os.listdir(resources_dir)
        if f.startswith("drugs_") and f.endswith(".parquet")
    ]

    if not drug_files:
        raise FileNotFoundError(
            f"No parquet files starting with 'drugs_' found in '{resources_dir}'"
        )

    logging.info(f"Found {len(drug_files)} drug files: {drug_files}")

    all_drugs = set()
    for file in drug_files:
        file_path = os.path.join(resources_dir, file)
        try:
            df = pd.read_parquet(file_path)
            # look for common drug column names
            drug_column = None
            for col in ["standard_concept_name", "drug_name", "drug", "concept_name"]:
                if col in df.columns:
                    drug_column = col
                    break

            if drug_column is None:
                logging.warning(f"No recognized drug column found in {file}, skipping")
                continue

            file_drugs = set(df[drug_column].dropna().astype(str).tolist())
            all_drugs.update(file_drugs)
            logging.info(
                f"Loaded {len(file_drugs)} drugs from {file} (column: {drug_column})"
            )

        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
            continue

    if not all_drugs:
        raise ValueError("No drugs found in any of the drug files")

    drugs_list = sorted(list(all_drugs))
    logging.info(f"Total unique drugs after deduplication: {len(drugs_list)}")

    return drugs_list


def get_processed_drugs(results_df: pd.DataFrame) -> Set[str]:
    """Get set of already processed drugs."""
    if results_df.empty:
        return set()
    return set(results_df["drug"].unique())


def estimate_probabilities_multi_seed(
    drugs: List[str],
    assessment_name: str,
    cot: bool,
    model_name: str,
    llm: LLM,
    sampling_params: SamplingParams,
    base_seed: int,
    max_retries: int = 20,
) -> pd.DataFrame:
    """
    Main estimation function using multi-seed retry approach.
    Process all drugs with one seed, collect failures, retry with offset seeds.
    """
    if not drugs:
        return pd.DataFrame()

    assessment_config = ASSESSMENT_CONFIGS.get(assessment_name)
    if not assessment_config:
        raise ValueError(f"Invalid assessment name: {assessment_name}")

    os.makedirs("results", exist_ok=True)
    model_shortname = model_name.split("/")[-1].lower()
    status_suffix = "_".join(filter(None, ["cot" if cot else "", f"seed{base_seed}"]))
    checkpoint_file = f"results/{assessment_name}_{model_shortname}{f'_{status_suffix}' if status_suffix else ''}.parquet"

    # load checkpoint with error handling
    results_df = pd.DataFrame()
    if os.path.exists(checkpoint_file):
        try:
            results_df = pd.read_parquet(checkpoint_file)
            logging.info(f"Loaded checkpoint with {len(results_df)} records.")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            results_df = pd.DataFrame()

    processed_drugs = get_processed_drugs(results_df)
    remaining_drugs = [drug for drug in drugs if drug not in processed_drugs]

    if not remaining_drugs:
        logging.info("All drugs already processed.")
        return results_df

    logging.info(
        f"Processing {len(remaining_drugs)} remaining drugs with multi-seed approach..."
    )

    batch_size = min(len(remaining_drugs), 1000)
    failed_drugs = set(remaining_drugs)

    for retry_round in range(max_retries):
        if not failed_drugs:
            break

        current_seed = base_seed + retry_round * 1000
        current_failed_list = list(failed_drugs)

        logging.info(
            f"Retry round {retry_round + 1}/{max_retries}: Processing {len(current_failed_list)} drugs with seed {current_seed}"
        )

        round_results = []
        round_failed = set()

        for i in tqdm(
            range(0, len(current_failed_list), batch_size),
            desc=f"Round {retry_round + 1} batches",
        ):
            batch_drugs = current_failed_list[i : i + batch_size]

            try:
                conversations = generate_batch_conversations(
                    batch_drugs, assessment_config, cot
                )

                current_sampling_params = SamplingParams(
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                    seed=current_seed,
                )

                outputs = llm.chat(
                    messages=conversations, sampling_params=current_sampling_params
                )

                batch_results, batch_failed = process_batch_results(
                    outputs, batch_drugs, current_seed
                )
                round_results.extend(batch_results)
                round_failed.update(batch_failed)

                logging.info(
                    f"Round {retry_round + 1}, Batch {i//batch_size + 1}: {len(batch_results)} successful, {len(batch_failed)} failed"
                )

                # checkpoint every 2 batches
                batch_idx = i // batch_size
                if (batch_idx + 1) % 2 == 0 and round_results:
                    results_df = save_checkpoint(
                        results_df, round_results, checkpoint_file
                    )
                    logging.info(
                        f"Checkpoint saved after batch {batch_idx + 1} with {len(results_df)} total records"
                    )
                    round_results = []

            except Exception as e:
                logging.error(
                    f"Error processing batch {i//batch_size + 1} in round {retry_round + 1}: {str(e)}"
                )
                round_failed.update(batch_drugs)
                continue

        failed_drugs = round_failed

        # final checkpoint for leftovers in this round
        if round_results:
            results_df = save_checkpoint(results_df, round_results, checkpoint_file)
            logging.info(
                f"Round {retry_round + 1} complete: {len(round_results)} new successes, {len(failed_drugs)} still failed. Total records: {len(results_df)}"
            )

        if not round_results:
            logging.warning(
                f"No successes in round {retry_round + 1}, trying with larger seed offset"
            )

    if failed_drugs:
        logging.warning(
            f"After {max_retries} rounds, {len(failed_drugs)} drugs permanently failed:"
        )
        for drug in sorted(failed_drugs):
            logging.warning(f"  - {drug}")

        permanent_failures = [
            {
                "drug": drug,
                "probability": None,
                "llm_response": f"FAILED_AFTER_{max_retries}_RETRY_ROUNDS",
                "seed": base_seed,
            }
            for drug in failed_drugs
        ]

        results_df = save_checkpoint(results_df, permanent_failures, checkpoint_file)

    final_df = save_checkpoint(results_df, [], checkpoint_file)
    logging.info(
        f"Multi-seed processing complete. Final results: {len(final_df)} total records"
    )

    return final_df


def main():
    global MODEL_NAME_GLOBAL
    parser = argparse.ArgumentParser(
        description="Estimate medical condition probabilities based on drugs."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Huggingface model name to use"
    )
    parser.add_argument(
        "--assessment",
        type=str,
        required=True,
        choices=list(ASSESSMENT_CONFIGS.keys()),
        help="Type of assessment to perform",
    )
    parser.add_argument(
        "--cot", action="store_true", help="Enable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature parameter for sampling",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only first 200 drugs and print responses)",
    )
    parser.add_argument(
        "--int4", action="store_true", help="Enable BitsAndBytes 4-bit quantization"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=20,
        help="Maximum number of retry rounds with different seeds",
    )

    args = parser.parse_args()

    MODEL_NAME_GLOBAL = args.model_name.lower()
    manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Starting estimation with configuration:")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Assessment: {args.assessment}")
    logging.info(f"Chain of thought: {args.cot}")
    logging.info(f"Number of GPUs: {args.num_gpus}")
    logging.info(f"Max retry rounds: {args.max_retries}")

    # initialize LLM with optimized settings
    llm_kwargs = {
        "model": args.model_name,
        "dtype": torch.bfloat16,
        "max_model_len": MAX_MODEL_LENGTH,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,  # fixed at 0.9
    }

    if args.num_gpus > 1:
        llm_kwargs["tensor_parallel_size"] = args.num_gpus

    if args.int4:
        llm_kwargs.update(
            {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}
        )

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {str(e)}")
        sys.exit(1)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=MAX_MODEL_LENGTH,
        seed=args.seed,
    )

    # load drug data from all parquet files in resources folder
    try:
        drugs = load_drug_data()
        logging.info(f"Loaded {len(drugs)} unique drugs from resources folder")
    except Exception as e:
        logging.error(f"Error loading drug data: {str(e)}")
        sys.exit(1)

    # debug mode processing
    if args.debug:
        logging.info("Debug mode enabled: processing only the first 5 drugs.")
        drugs = drugs[:5]

    # start processing
    start_time = time.time()
    results_df = estimate_probabilities_multi_seed(
        drugs=drugs,
        assessment_name=args.assessment,
        cot=args.cot,
        model_name=args.model_name,
        llm=llm,
        sampling_params=sampling_params,
        base_seed=args.seed,
        max_retries=args.max_retries,
    )

    end_time = time.time()
    processing_time = end_time - start_time

    logging.info(f"Estimation complete. Final dataset shape: {results_df.shape}")
    logging.info(f"Total processing time: {processing_time:.2f} seconds")
    logging.info(f"Average time per drug: {processing_time/len(drugs):.2f} seconds")

    # debug mode output
    if args.debug:
        debug_subset = results_df.head(200)
        for idx, row in debug_subset.iterrows():
            logging.info(f"Drug: {row['drug']}")
            logging.info(f"Response: {row['llm_response'][:200]}...")
            logging.info(f"Probability: {row['probability']}")
            logging.info(f"{'-'*40}")


if __name__ == "__main__":
    main()
