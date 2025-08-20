# train_nemotron_lora.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

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

# Let HF Transformers handle placement automatically
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",     # avoids manual .to() issues
    torch_dtype="auto"     # match precision automatically
)

# -----------------------------
# 3. Tokenize dataset
# -----------------------------
def tokenize(batch):
    # combine prompt + completion into single string
    texts = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# -----------------------------
# 4. LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # common for Mistral-like models
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 5. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora_nemotron",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,  # <-- critical fix
    report_to="none"  # disable wandb unless you want logging
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -----------------------------
# 7. Start training
# -----------------------------
trainer.train()
