import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from Hugging Face
dataset = load_dataset("shaneperry0101/health-chatbot")

# Convert to Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Select necessary columns and concatenate for training
df["text"] = df["prompt"] + " " + df["response"]

# Split into 80% train, 20% test
train_texts, test_texts = train_test_split(df["text"], test_size=0.2, random_state=42)

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Assign a padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="longest",  
        max_length=128
    )

# Convert texts into Hugging Face datasets
train_dataset = Dataset.from_dict({"text": train_texts.tolist()}).map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = Dataset.from_dict({"text": test_texts.tolist()}).map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator (for batching)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"  
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train model
trainer.train()

# Evaluate model
results = trainer.evaluate()
print("Test Evaluation Results:", results)

# Save fine-tuned model
model.save_pretrained("./fine_tuned_distilgpt2")
tokenizer.save_pretrained("./fine_tuned_distilgpt2")
