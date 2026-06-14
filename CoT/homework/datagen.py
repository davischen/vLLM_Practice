import os
import json
import math
from tqdm import tqdm

# Prevent OOM in macOS MPS environment
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def generate_dataset(
    output_json: str, 
    oversample: int = 10, 
    temperature: float = 0.6, 
    max_retries: int = 3  
):
    from .cot import CoTModel
    from .data import is_answer_valid, Dataset

    # 1. Load the model (Change checkpoint here if needed)
    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model = CoTModel(checkpoint=checkpoint)

    # 2. Prepare training data
    trainset = Dataset("train")
    
    # Extract dataset contents (Using list comprehension for cleaner syntax)
    questions = [item[0] for item in trainset]
    answers = [item[1] for item in trainset]
    prompts = [model.format_prompt(q) for q in questions]

    print(f"🚀 Starting dataset generation with {len(prompts)} prompts. Max retries: {max_retries}")

    best_samples = []
    chunk_size = 3  # Adjust based on hardware (Mac=2~3, 3090=5+)

    # 3. Chunking to control memory usage
    for idx in tqdm(range(0, len(prompts), chunk_size), desc="Generating in chunks"):
        current_questions = questions[idx : idx + chunk_size]
        current_prompts = prompts[idx : idx + chunk_size]
        current_answers = answers[idx : idx + chunk_size]

        # Keep track of indices within the current chunk that still need a valid answer
        unresolved_indices = list(range(len(current_questions)))
        attempts = 0

        # Smart Retry Loop: Only retry the prompts that failed
        while unresolved_indices and attempts < max_retries:
            attempts += 1
            
            # Prepare inputs strictly for unresolved prompts
            active_prompts = [current_prompts[i] for i in unresolved_indices]
            
            # Batch generate multiple responses (Oversampling)
            generations = model.batched_generate(
                active_prompts, 
                num_return_sequences=oversample, 
                temperature=temperature
            )

            still_unresolved = []
            
            # 4. Validate and select successful samples
            for active_idx, generated_group in enumerate(generations):
                original_idx = unresolved_indices[active_idx]
                question = current_questions[original_idx]
                expected_answer = current_answers[original_idx]
                
                solved = False
                for sample in generated_group:
                    sample_answer = model.parse_answer(sample)

                    # Safety check for NaN and correctness validation
                    if not math.isnan(sample_answer) and is_answer_valid(sample_answer, float(expected_answer)):
                        # If correct, save as a successful training sample
                        best_samples.append([question, expected_answer, sample])
                        solved = True
                        break # One correct reasoning trace is enough
                
                # If no valid answer was found in this oversample batch, queue for retry
                if not solved:
                    still_unresolved.append(original_idx)
            
            # Update unresolved list for the next retry iteration
            unresolved_indices = still_unresolved

    print(f"✅ Generation complete! Accepted {len(best_samples)} / {len(prompts)} valid samples.")

    # 5. Save to JSON file ensuring correct text encoding
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_samples, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)