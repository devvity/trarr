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
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding works

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# -----------------------------
# 3. Tokenize dataset
# -----------------------------
def tokenize(example):
    # Your dataset already has "prompt" and "completion"
    text = example["prompt"] + example["completion"]
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=False)

# -----------------------------
# 4. LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # safe choice for Mistral
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 5. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora_mistral",
    per_device_train_batch_size=2,  # reduce if OOM
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    fp16=True,
    save_total_limit=2
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer  # keep tokenizer for generation later
)

# -----------------------------
# 7. Start training
# -----------------------------
trainer.train()
