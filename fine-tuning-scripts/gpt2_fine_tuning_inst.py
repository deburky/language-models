import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# ----- Data -----

# Load a very small dataset (only 100 examples)
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:100]")
dataset = dataset.train_test_split(test_size=0.1)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ----- Model -----

# Use a very small model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix padding issue
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to("mps")


# Format input correctly
def formatting_prompts_func(example):
    return (
        f"### Instruction: {example['instruction']}\n### Response: {example['output']}"
    )


# Set up collator for completions only
response_template = "### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Define Training Arguments (Reduce Batch Size + Max Length)
training_args = TrainingArguments(
    output_dir="./results_distilgpt2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,  # Reduce batch size (helps MPS OOM)
    per_device_eval_batch_size=4,  # Reduce batch size for eval
    num_train_epochs=1,  # Just 1 epoch for testing
    gradient_accumulation_steps=4,  # Helps with small batches
    max_steps=50,  # Train for only 50 steps
    weight_decay=0.01,
    logging_dir="./logs_distilgpt2",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=False,  # Don't load best model to save memory
    fp16=torch.cuda.is_available(),  # Use fp16 if CUDA is available
    bf16=torch.backends.mps.is_available(),  # Use bf16 if using Apple Silicon (MPS)
)

# Initialize Trainer with eval_dataset
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args=training_args,
)

# Start Training
trainer.train()

# Save model and tokenizer to /results_distilgpt2
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# ----- Evaluation -----

def generate_response(instruction, max_length=512):
    """Generate a response using the fine-tuned model."""
    input_text = f"### Instruction: {instruction}\n### Response:"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {
        key: val.to(model.device) for key, val in inputs.items()
    }  # Move to device

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(input_text, "").strip()  # Remove input from output


# ✅ Test the fine-tuned model
test_instruction = "Write a function for fibonacci in python."
response = generate_response(test_instruction)
print(f"🚀 Model Response:\n{response}")
