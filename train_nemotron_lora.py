# train_nemotron_lora.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -----------------------------
# 1. Load dataset (JSONL format)
# -----------------------------
dataset = load_dataset(
    "json",
    data_files="/workspace/single_person_dataset.jsonl",
    split="train"
)

# -----------------------------
# 2. Load tokenizer and model
# -----------------------------
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # important for causal LM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# -----------------------------
# 3. LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # works well for Mistral
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 4. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora_mistral",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,  # important for TRL
    report_to="none"
)

# -----------------------------
# 5. SFT Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    max_seq_length=128,  # truncate longer samples
    dataset_text_field=None,  # we’ll pass combined text manually
    formatting_func=lambda ex: ex["prompt"] + " " + ex["completion"]
)

# -----------------------------
# 6. Start training
# -----------------------------
trainer.train()
