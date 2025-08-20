# train_nemotron_lora.py  (version: robust SFTTrainer)
import inspect
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -----------------------------
# 1) Load dataset (JSONL)
# -----------------------------
DATA_PATH = "/workspace/single_person_dataset.jsonl"
ds = load_dataset("json", data_files=DATA_PATH, split="train")

# -----------------------------
# 2) Base model & tokenizer
# -----------------------------
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

# -----------------------------
# 3) LoRA config (Mistral)
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # lightweight & proven for Mistral/LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 4) Training args
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora_mistral",
    per_device_train_batch_size=4,       # keep small; use grad accum to simulate bigger
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    fp16=True,                           # Ampere supports this; safer than bf16 across envs
    save_total_limit=2,
    remove_unused_columns=False,         # TRL expects raw cols to be present
    gradient_checkpointing=True,         # reduce VRAM
    report_to="none",
)

# -----------------------------
# 5) formatting func (batch -> list[str])
# -----------------------------
def formatting_func(batch):
    # TRL expects a list[str] for the batch
    return [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]

# -----------------------------
# 6) Build SFTTrainer with version-agnostic kwargs
# -----------------------------
sig = inspect.signature(SFTTrainer.__init__)
param_names = set(sig.parameters.keys())

sft_kwargs = dict(
    model=model,
    train_dataset=ds,
    args=training_args,
    peft_config=lora_config,
    max_seq_length=128,
    formatting_func=formatting_func,
)

# Newer TRL removed "tokenizer" and sometimes wants "processing_class"
if "processing_class" in param_names:
    sft_kwargs["processing_class"] = tokenizer
elif "tokenizer" in param_names:
    sft_kwargs["tokenizer"] = tokenizer
# else: neither is required; TRL will infer from model

trainer = SFTTrainer(**sft_kwargs)

# -----------------------------
# 7) Train
# -----------------------------
trainer.train()
