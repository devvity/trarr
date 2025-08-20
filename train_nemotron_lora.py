# train_nemotron_lora.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------
# 1. Load dataset
# -----------------------------
dataset = load_dataset("json", data_files="/workspace/single_person_dataset.jsonl")

# -----------------------------
# 2. Load tokenizer and model
# -----------------------------
model_name = "nvidia/Nemotron-Nano-9B-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # important for causal LM
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# -----------------------------
# 3. Tokenize dataset
# -----------------------------
def tokenize(example):
    return tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# -----------------------------
# 4. LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # typical for causal LM
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
    per_device_train_batch_size=4,  # lower for GPU memory safety
    gradient_accumulation_steps=4,  # simulates larger batch
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    fp16=True,  # use mixed precision
    save_total_limit=2
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# -----------------------------
# 7. Start training
# -----------------------------
trainer.train()
