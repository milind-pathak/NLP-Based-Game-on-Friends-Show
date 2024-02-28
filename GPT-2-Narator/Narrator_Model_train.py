from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Datasets/Narrator_Scene_Description_Dataset.csv')

# Concatenate the text fields into a single text column for training
data['narration'] = data.apply(lambda r: f"{r['Player Action']} {r['Narration']} {r['Scene Description']}", axis=1)

# Write the text to a file
with open('narration_dataset.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(data['narration'].tolist()))

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize the text data
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='narration_dataset.txt',
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Set the training arguments
training_args = TrainingArguments(
    output_dir="/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/gpt2-finetuned-narrator",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/gpt2-finetuned-narrator")
tokenizer.save_pretrained("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/gpt2-finetuned-narrator")