from .base_llm import BaseLLM
from .sft import test_model, tokenize
from pathlib import Path
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import json

from .data import Dataset, benchmark

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format RFT example - the reasoning already contains the answer tag.
    """
    return {
        "question": question,
        "answer": reasoning  # The reasoning already includes <answer>...</answer>
    }

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)

def format_example(prompt: str, answer: str, reasoning: str = None) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # raise NotImplementedError()
    if reasoning is not None:
        # Use reasoning as the model's target (RFT)
        return {"question": prompt, "answer": reasoning}
    else:
        # For SFT data: only train to produce <answer>...</answer>
        formatted_answer = f"<answer>{round(float(answer), 4)}</answer>"
        return {"question": prompt, "answer": formatted_answer}

def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()
    # raise NotImplementedError()
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from .base_llm import BaseLLM
    #from .sft import TokenizedDataset, format_example
    from .data import Dataset
    import json
    from pathlib import Path
    
    # Load base model
    llm = BaseLLM()
    
    # Configure LoRA with slightly larger rank r for RFT and alpha=4r
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        #lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    
    rft_dataset = Dataset("rft")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_dataset, format_example)
    
    # Training arguments
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=1e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
    )
    
    # Create trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})