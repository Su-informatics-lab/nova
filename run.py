"""
This script estimates the probability of various medical conditions and patient
characteristics based on medication data using LLMs.

Supported Assessments:
- Binary: Type II diabetes, AUDIT-C, insurance status, alcohol abuse
- Ordinal (5 levels): Fatigue and anxiety

Usage:
    python estimate_prob_given_drug.py --model_name MODEL_NAME \
    --assessment ASSESSMENT_TYPE [options]

Required Arguments:
    --model_name     Huggingface model name (e.g., meta-llama/Llama-3.1-70B-Instruct)
    --assessment     Assessment type (diabetes|audit_c|fatigue|anxiety|insurance|alcohol_abuse)

Optional Arguments:
    --cot           Enable chain-of-thought reasoning
    --enforce       Enforce LLMs to provide estimation even uncertain
    --num_gpus      Number of GPUs for tensor parallelism (default: 1)
    --temperature   Sampling temperature (default: 0.6)
    --input_file    Input parquet file with drug names
        (default: resources/drug_15980.parquet)
    --seed          Global random seed for reproducibility (default: 42)
    --debug         Enable debug mode (process only first 5 drugs and print responses)
    --max_concurrent_requests  Maximum concurrent requests to vLLM (default: auto)
    --checkpoint_interval     How often to save checkpoints (default: 50)

Output:
    Generates a parquet file containing:
    - Drug names
    - Estimated probabilities
    - Full LLM responses
    - Probabilities for each severity level (only for ordinal assessments)
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

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

class QueryType(Enum):
    BINARY = "binary"
    ORDINAL = "ordinal"

@dataclass
class AssessmentConfig:
    name: str
    query_type: QueryType
    system_prompt: str
    question: str
    levels: Optional[List[str]] = None

    def create_prompt(self, drug: str, level: Optional[str] = None, cot: bool = False) -> str:
        """
        Create a simple, direct prompt for the assessment.
        Args:
            drug: Name of the drug
            level: Level for ordinal assessments (optional)
            cot: Whether to include chain-of-thought reasoning instruction
        """
        if self.query_type == QueryType.BINARY:
            questionnaire_info = f"\n\n{self.question}\n\n" if self.question else "\n\n"
            base_prompt = (
                f"Given that a patient took {drug}, estimate the probability that they have {self.name}."
                f"{questionnaire_info}"
                "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
            )
        else:
            if self.question:
                base_prompt = (
                    f"For a patient taking {drug}, what is the probability they would report '{level}'?\n\n"
                    "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
                )
            else:
                base_prompt = (
                    f"For a patient taking {drug}, estimate the probability of {level}. "
                    "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
                )

        if cot:
            base_prompt += "\nYou may think aloud and reason step-by-step before reaching the final answer."

        return base_prompt

    def get_system_prompt(self, enforce: bool = False) -> str:
        """
        Get system prompt with optional enforcement language.
        Args:
            enforce: Whether to add enforcement language
        """
        base_system_prompt = self.system_prompt
        if enforce:
            enforcement_addition = (
                " You must always provide a numerical probability estimate between 0 and 1, "
                "even if uncertain. If you are unsure, provide your best estimate based on "
                "available knowledge. You cannot refuse to provide an estimate."
            )
            base_system_prompt += enforcement_addition
        return base_system_prompt

ASSESSMENT_CONFIGS = {
    "diabetes": AssessmentConfig(
        name="Type II diabetes",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has Type II diabetes based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "audit_c": AssessmentConfig(
        name="high-risk AUDIT-C score (4+ for men, 3+ for women)",
        query_type=QueryType.BINARY,
        question="""The AUDIT-C score is based on three questions:

        1. How often do you have a drink containing alcohol?
           a. Never (0 points)
           b. Monthly or less (1 point)
           c. 2-4 times a month (2 points)
           d. 2-3 times a week (3 points)
           e. 4 or more times a week (4 points)

        2. How many standard drinks containing alcohol do you have on a typical day?
           a. 1 or 2 (0 points)
           b. 3 or 4 (1 point)
           c. 5 or 6 (2 points)
           d. 7 to 9 (3 points)
           e. 10 or more (4 points)

        3. How often do you have six or more drinks on one occasion?
           a. Never (0 points)
           b. Less than monthly (1 point)
           c. Monthly (2 points)
           d. Weekly (3 points)
           e. Daily or almost daily (4 points)

        Total score ranges from 0-12. For men, a score of 4+ indicates high-risk drinking.
        For women, a score of 3+ indicates high-risk drinking.""",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has a high-risk AUDIT-C score based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "audit_c_simplified": AssessmentConfig(
        name="high-risk AUDIT-C score (4+ for men, 3+ for women)",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has a high-risk AUDIT-C score based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "fatigue": AssessmentConfig(
        name="fatigue level",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how would you rate your fatigue?",
        levels=["None", "Mild", "Moderate", "Severe", "Very Severe"],
        system_prompt=(
             "You are a medical language model designed to estimate the probability of different fatigue levels a patient has based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "anxiety": AssessmentConfig(
        name="emotional problems",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how often have you been bothered by emotional problems such as feeling anxious, depressed or irritable?",
        levels=["Never", "Rarely", "Sometimes", "Often", "Always"],
        system_prompt=(
            "You are a medical language model designed to estimate the probability of different frequencies of emotional problems based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "insurance": AssessmentConfig(
        name="employer-based insurance",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has employer-based insurance based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "alcohol_abuse": AssessmentConfig(
        name="alcohol abuse",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has alcohol abuse based on the specific medicine they use. "
            "For this task, refer to the following ICD-10 codes as definitions for alcohol abuse:\n\n"
            "F10 - Alcohol-related disorders:\n"
            "  F10.1 Alcohol abuse (F10.10 Uncomplicated, F10.11 In remission, F10.12 With intoxication (uncomplicated, delirium, unspecified), "
            "F10.13 With withdrawal (uncomplicated, delirium, perceptual disturbance, unspecified), F10.14 With alcohol-induced mood disorder, "
            "F10.15 With alcohol-induced psychotic disorder (delusions, hallucinations, unspecified), F10.18 With other alcohol-induced disorders "
            "(anxiety, sexual dysfunction, sleep disorder, other), F10.19 With unspecified alcohol-induced disorder)\n"
            "  F10.2 Alcohol dependence (F10.20 Uncomplicated, F10.21 In remission, F10.22 With intoxication (uncomplicated, delirium, unspecified), "
            "F10.23 With withdrawal (uncomplicated, delirium, perceptual disturbance, unspecified), F10.24 With alcohol-induced mood disorder, "
            "F10.25 With alcohol-induced psychotic disorder (delusions, hallucinations, unspecified), F10.26 With alcohol-induced persisting amnestic disorder, "
            "F10.27 With alcohol-induced persisting dementia, F10.28 With other alcohol-induced disorders (anxiety, sexual dysfunction, sleep disorder, other), "
            "F10.29 With unspecified alcohol-induced disorder)\n"
            "E52 - Niacin deficiency (pellagra)\n"
            "G62.1 - Alcoholic polyneuropathy\n"
            "I42.6 - Alcoholic cardiomyopathy\n"
            "K29.2 - Alcoholic gastritis\n"
            "K70 - Alcoholic liver disease (K70.0 Fatty liver, K70.3 Cirrhosis of liver, K70.9 Unspecified)\n"
            "T51 - Toxic effect of alcohol (T51.0 Ethanol, T51.1 Methanol, T51.2 2-Propanol, T51.3 Fusel oil, T51.8 Other alcohols, T51.9 Unspecified alcohol)\n"
            "Z50.2 - Alcohol rehabilitation\n"
            "Z71.4 - Alcohol abuse counseling and surveillance\n"
            "Z72.1 - Alcohol use\n\n"
            "Estimate the probability that a patient has alcohol abuse based solely on the medication data provided. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )
    ),
    "breast_cancer": AssessmentConfig(
        name="breast cancer",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a woman has breast cancer based on the provided diagnostic codes and medication data. "
            "For this task, refer to the following ICD codes and their definitions as criteria for breast cancer:\n\n"
            "Female Breast Cancer Diagnosis Codes:\n"
            "  ICD-10CM:\n"
            "    C50.01* - Malignant neoplasm of nipple and areola, female "
            "(includes C50.01, C50.011, C50.012, C50.019)\n"
            "    C50.1*  - Malignant neoplasm of central portion of breast "
            "(includes C50.1, C50.11, C50.111, C50.112, C50.119)\n"
            "    C50.2*  - Malignant neoplasm of upper-inner quadrant of breast "
            "(includes C50.2, C50.21, C50.211, C50.212, C50.219)\n"
            "    C50.3*  - Malignant neoplasm of lower-inner quadrant of breast "
            "(includes C50.3, C50.31, C50.311, C50.312, C50.319)\n"
            "    C50.4*  - Malignant neoplasm of upper-outer quadrant of breast "
            "(includes C50.4, C50.41, C50.411, C50.412, C50.419)\n"
            "    C50.5*  - Malignant neoplasm of lower-outer quadrant of breast "
            "(includes C50.5, C50.51, C50.511, C50.512, C50.519)\n"
            "    C50.6*  - Malignant neoplasm of axillary tail of breast "
            "(includes C50.6, C50.61, C50.611, C50.612, C50.619)\n"
            "    C50.81* - Malignant neoplasm of overlapping sites of breast, female "
            "(includes C50.81, C50.811, C50.812, C50.819)\n"
            "    C50.91* - Malignant neoplasm of breast of unspecified site, female "
            "(includes C50.91, C50.911, C50.912, C50.919)\n"
            "    D05*    - Carcinoma in situ of breast "
            "(includes D05, D05.1, D05.10, D05.11, D05.12, D05.8, D05.80, D05.81, D05.82, D05.9, D05.90, D05.91, D05.92)\n"
            "  ICD-9CM:\n"
            "    174*    - Malignant neoplasm of female breast "
            "(includes 174, 174.0, 174.1, 174.2, 174.3, 174.4, 174.5, 174.6, 174.8, 174.9)\n"
            "    233.0   - Carcinoma in situ of breast\n\n"
            "Breast Cancer History Codes:\n"
            "  ICD-10CM:\n"
            "    Z85.3   - Personal history of malignant neoplasm of breast\n"
            "    Z86.000 - Personal history of in-situ neoplasm of breast\n"
            "  ICD-9CM:\n"
            "    V10.3   - Personal history of malignant neoplasm of breast\n\n"
            "Estimate the probability that a woman has breast cancer based solely on the provided "
            "medication data and diagnostic codes. Provide the probability enclosed within "
            "[ESTIMATION] and [/ESTIMATION] tags."
        )
    ),
    "breast_cancer_simplified": AssessmentConfig(
        name="breast cancer",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a woman has breast cancer based solely on medication data. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )
    ),
    "alcoholic_hepatitis_simplified": AssessmentConfig(
        name="alcoholic hepatitis",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has alcoholic hepatitis based solely on medication data. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )
    ),
    "alcohol_abuse_simplified": AssessmentConfig(
        name="alcohol abuse",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has alcohol abuse based solely on medication data. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )
    ),
    "hypertension_simplified": AssessmentConfig(
        name="hypertension",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has hypertension based solely on medication data. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )
    )
}

def create_conversation(
        drug: str, assessment_config: AssessmentConfig, level: Optional[str], cot: bool,
        enforce: bool = False
) -> List[Dict]:
    """
    Create a conversation template.
    For models that do not support a separate system role (e.g., DeepSeek-R1 and Gemma-2),
    prepend the system instruction to the user prompt.
    """
    system_prompt = assessment_config.get_system_prompt(enforce)
    # get system prompt; if the model is deepseek-r1, add extra directive for chain-of-thought
    # refer to https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    if "deepseek-r1" in MODEL_NAME_GLOBAL:
        system_prompt += "\nPlease ensure that your answer begins with \"<think>\n\"."
    user_prompt = assessment_config.create_prompt(drug, level, cot)
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
    tag_match = re.search(r'\[ESTIMATION\](.*?)\[/ESTIMATION\]', response_text, re.DOTALL)
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
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', estimation_text)
        if percentage_match:
            try:
                value = float(percentage_match.group(1)) / 100
                if np.isfinite(value) and 0 <= value <= 1:
                    return value
            except ValueError:
                pass
    return None

def generate_batch_conversations(
    drug_level_pairs: List[Tuple[str, Optional[str]]],
    assessment_config: AssessmentConfig,
    cot: bool,
    enforce: bool
) -> List[List[Dict]]:
    """Generate batch conversations for vLLM processing."""
    conversations = []
    for drug, level in drug_level_pairs:
        conversation = create_conversation(drug, assessment_config, level, cot, enforce)
        conversations.append(conversation)
    return conversations

def process_batch_results(
    outputs: List,
    drug_level_pairs: List[Tuple[str, Optional[str]]],
    global_seed: int
) -> Tuple[List[Dict], List[Tuple[str, Optional[str]]]]:
    """
    Process batch results from vLLM.
    Returns:
        - List of successful results
        - List of failed drug_level_pairs that need retry
    """
    results = []
    failed_pairs = []

    for i, ((drug, level), output) in enumerate(zip(drug_level_pairs, outputs)):
        try:
            if output and output.outputs and len(output.outputs) > 0:
                response_text = output.outputs[0].text
                probability = extract_probability(response_text)

                result = {
                    "drug": drug,
                    "probability": probability,
                    "llm_response": response_text,
                    "seed": global_seed,
                }
                if level is not None:
                    result["level"] = level
                results.append(result)

                # log if probability extraction failed
                if probability is None:
                    logging.warning(f"Failed to extract probability for {drug}{f' (level: {level})' if level else ''}")
            else:
                logging.warning(f"Empty output for {drug}{f' (level: {level})' if level else ''}, marking for retry")
                failed_pairs.append((drug, level))
        except Exception as e:
            logging.error(f"Error processing result for {drug}{f' (level: {level})' if level else ''}: {str(e)}")
            failed_pairs.append((drug, level))

    return results, failed_pairs

def estimate_probabilities_batch(
        drugs: List[str],
        assessment_name: str,
        cot: bool,
        enforce: bool,
        model_name: str,
        llm: LLM,
        sampling_params: SamplingParams,
        global_seed: int,
        checkpoint_interval: int = 50,
        max_concurrent_requests: Optional[int] = None,
        max_retries: int = 3,
) -> pd.DataFrame:
    """Main estimation function using vLLM batch processing with individual retry logic."""
    if not drugs:
        return pd.DataFrame()

    assessment_config = ASSESSMENT_CONFIGS.get(assessment_name)
    if not assessment_config:
        raise ValueError(f"Invalid assessment name: {assessment_name}")

    os.makedirs("results", exist_ok=True)
    model_shortname = model_name.split('/')[-1].lower()
    status_suffix = '_'.join(filter(None, [
        'cot' if cot else '',
        'enforce' if enforce else '',
        f'seed{global_seed}'
    ]))
    checkpoint_file = f"results/{assessment_name}_{model_shortname}{f'_{status_suffix}' if status_suffix else ''}.parquet"

    # load checkpoint with error handling
    results_df = pd.DataFrame()
    if os.path.exists(checkpoint_file):
        try:
            results_df = pd.read_parquet(checkpoint_file)
            if assessment_config.query_type == QueryType.BINARY:
                processed_drugs = set(results_df["drug"].unique())
            else:
                # for ordinal, check if all levels are processed for each drug
                levels = assessment_config.levels
                processed_pairs = set(zip(results_df["drug"], results_df["level"]))
                processed_drugs = set()
                for drug in drugs:
                    if all((drug, level) in processed_pairs for level in levels):
                        processed_drugs.add(drug)

            remaining_drugs = [d for d in drugs if d not in processed_drugs]
            logging.info(f"Loaded checkpoint with {len(results_df)} records. {len(remaining_drugs)} drugs remaining.")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            remaining_drugs = drugs
    else:
        remaining_drugs = drugs

    if not remaining_drugs:
        logging.info("All drugs already processed.")
        return results_df

    # prepare drug-level pairs for processing
    levels = [None] if assessment_config.query_type == QueryType.BINARY else assessment_config.levels
    drug_level_pairs = []
    for drug in remaining_drugs:
        for level in levels:
            drug_level_pairs.append((drug, level))

    logging.info(f"Processing {len(drug_level_pairs)} drug-level combinations...")

    all_results = []
    failed_pairs = []  # track failed pairs for retry

    # use max_concurrent_requests if specified, otherwise let vLLM decide
    batch_size = max_concurrent_requests if max_concurrent_requests else len(drug_level_pairs)

    # process in batches for memory management and checkpointing
    for i in tqdm(range(0, len(drug_level_pairs), batch_size), desc="Processing batches"):
        batch_pairs = drug_level_pairs[i:i + batch_size]

        try:
            # generate conversations for the batch
            conversations = generate_batch_conversations(
                batch_pairs, assessment_config, cot, enforce
            )

            # process batch with vLLM
            outputs = llm.chat(messages=conversations, sampling_params=sampling_params)

            # process results and identify failures
            batch_results, batch_failed = process_batch_results(outputs, batch_pairs, global_seed)
            all_results.extend(batch_results)
            failed_pairs.extend(batch_failed)

            logging.info(f"Batch {i//batch_size + 1}: {len(batch_results)} successful, {len(batch_failed)} failed")

        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # if entire batch fails, mark all pairs for retry
            failed_pairs.extend(batch_pairs)
            continue

        # save checkpoint periodically
        if (i // batch_size + 1) % checkpoint_interval == 0:
            try:
                temp_df = pd.concat([results_df, pd.DataFrame(all_results)], ignore_index=True)
                temp_df.to_parquet(checkpoint_file)
                logging.info(f"Checkpoint saved with {len(temp_df)} total records")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {str(e)}")

    # retry failed pairs individually
    if failed_pairs:
        logging.info(f"Retrying {len(failed_pairs)} failed drug-level combinations individually...")
        retry_results = retry_failed_pairs(
            failed_pairs, assessment_config, cot, enforce, llm, sampling_params,
            global_seed, max_retries
        )
        all_results.extend(retry_results)

    # save final results
    try:
        final_df = pd.concat([results_df, pd.DataFrame(all_results)], ignore_index=True)
        final_df.to_parquet(checkpoint_file)
        logging.info(f"Final results saved with {len(final_df)} total records")
    except Exception as e:
        logging.error(f"Error saving final results: {str(e)}")
        final_df = pd.concat([results_df, pd.DataFrame(all_results)], ignore_index=True)

    return final_df

def save_checkpoint(results_df: pd.DataFrame, new_results: List[Dict], checkpoint_file: str) -> pd.DataFrame:
    """Save checkpoint and return updated dataframe."""
    try:
        if new_results:
            updated_df = pd.concat([results_df, pd.DataFrame(new_results)], ignore_index=True)
        else:
            updated_df = results_df
        updated_df.to_parquet(checkpoint_file)
        return updated_df
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        return results_df

def get_processed_pairs(results_df: pd.DataFrame, assessment_config: AssessmentConfig) -> set:
    """Get set of already processed drug-level pairs."""
    if results_df.empty:
        return set()

    if assessment_config.query_type == QueryType.BINARY:
        # for binary, just need drug names
        return set((drug, None) for drug in results_df["drug"].unique())
    else:
        # for ordinal, need drug-level pairs
        return set(zip(results_df["drug"], results_df["level"]))

def retry_failed_pairs_batch(
) -> List[Dict]:
    """Retry failed pairs individually with exponential backoff."""
    results = []

    for drug, level in tqdm(failed_pairs, desc="Retrying failed pairs"):
        success = False
        for attempt in range(max_retries):
            try:
                conversation = create_conversation(drug, assessment_config, level, cot, enforce)

                # use different seed for each retry attempt
                retry_params = SamplingParams(
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                    seed=global_seed + attempt + 1000  # Offset to avoid collision
                )

                output = llm.chat(messages=[conversation], sampling_params=retry_params)[0]

                if output and output.outputs and len(output.outputs) > 0:
                    response_text = output.outputs[0].text
                    probability = extract_probability(response_text)

                    result = {
                        "drug": drug,
                        "probability": probability,
                        "llm_response": response_text,
                        "seed": global_seed + attempt + 1000,
                    }
                    if level is not None:
                        result["level"] = level

                    results.append(result)
                    success = True
                    break
                else:
                    logging.warning(f"Retry {attempt + 1}/{max_retries} failed for {drug}{f' (level: {level})' if level else ''}: Empty output")

            except Exception as e:
                logging.warning(f"Retry {attempt + 1}/{max_retries} failed for {drug}{f' (level: {level})' if level else ''}: {str(e)}")
                continue

        if not success:
            # create a result with None probability to track the failure
            result = {
                "drug": drug,
                "probability": None,
                "llm_response": f"FAILED_AFTER_{max_retries}_RETRIES",
                "seed": global_seed,
            }
            if level is not None:
                result["level"] = level
            results.append(result)
            logging.error(f"Permanently failed after {max_retries} retries: {drug}{f' (level: {level})' if level else ''}")

    return results

def main():
    global MODEL_NAME_GLOBAL
    parser = argparse.ArgumentParser(
        description="Estimate medical condition probabilities based on drugs."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Huggingface model name to use")
    parser.add_argument("--assessment", type=str, required=True,
                        choices=list(ASSESSMENT_CONFIGS.keys()),
                        help="Type of assessment to perform")
    parser.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought reasoning")
    parser.add_argument("--enforce", action="store_true",
                        help="Enforce LLMs to provide estimation even when uncertain")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature parameter for sampling")
    parser.add_argument("--input_file", type=str,
                        default="resources/drugs_15980.parquet",
                        help="Input file containing drug names")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (process only first 5 drugs and print responses)")
    parser.add_argument("--int4", action="store_true",
                        help="Enable BitsAndBytes 4-bit quantization")
    parser.add_argument("--max_concurrent_requests", type=int, default=None,
                        help="Maximum concurrent requests to vLLM (default: let vLLM decide)")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="How often to save checkpoints (default: 50 batches)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio (default: 0.9)")
    parser.add_argument("--max_retries", type=int, default=10,
                        help="Maximum number of retries for failed items (default: 3)")

    args = parser.parse_args()

    MODEL_NAME_GLOBAL = args.model_name.lower()
    manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Starting estimation with configuration:")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Assessment: {args.assessment}")
    logging.info(f"Chain of thought: {args.cot}")
    logging.info(f"Enforce: {args.enforce}")
    logging.info(f"Quantization: {args.int4}")
    logging.info(f"Number of GPUs: {args.num_gpus}")
    logging.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
    logging.info(f"Max concurrent requests: {args.max_concurrent_requests or 'auto'}")

    # initialize LLM with optimized settings
    llm_kwargs = {
        "model": args.model_name,
        "dtype": torch.bfloat16,
        "max_model_len": MAX_MODEL_LENGTH,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    if args.num_gpus > 1:
        llm_kwargs["tensor_parallel_size"] = args.num_gpus

    if args.int4:
        llm_kwargs.update({
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes"
        })

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {str(e)}")
        sys.exit(1)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=MAX_MODEL_LENGTH,
        seed=args.seed
    )

    # load drug data
    try:
        df = pd.read_parquet(args.input_file)
        drugs = df["standard_concept_name"].tolist()
        logging.info(f"Loaded {len(drugs)} drugs from {args.input_file}")
    except Exception as e:
        logging.error(f"Error loading input file: {str(e)}")
        sys.exit(1)

    # debug mode processing
    if args.debug:
        logging.info("Debug mode enabled: processing only the first 5 drugs.")
        drugs = drugs[:5]

    # start processing
    start_time = time.time()
    results_df = estimate_probabilities_batch(
        drugs=drugs,
        assessment_name=args.assessment,
        cot=args.cot,
        enforce=args.enforce,
        model_name=args.model_name,
        llm=llm,
        sampling_params=sampling_params,
        global_seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        max_concurrent_requests=args.max_concurrent_requests,
        max_retries=args.max_retries,
    )

    end_time = time.time()
    processing_time = end_time - start_time

    logging.info(f"Estimation complete. Final dataset shape: {results_df.shape}")
    logging.info(f"Total processing time: {processing_time:.2f} seconds")
    logging.info(f"Average time per drug: {processing_time/len(drugs):.2f} seconds")

    # debug mode output
    if args.debug:
        debug_subset = results_df.head(10)
        for idx, row in debug_subset.iterrows():
            logging.info(f"Drug: {row['drug']}")
            if 'level' in row:
                logging.info(f"Level: {row['level']}")
            logging.info(f"Response: {row['llm_response'][:200]}...")
            logging.info(f"Probability: {row['probability']}")
            logging.info(f"{'-'*40}")

if __name__ == "__main__":
    main()
