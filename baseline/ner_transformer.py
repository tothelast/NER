from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import numpy as np
from evaluate import load

# Input data
tokens = [
    ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
    ["Elon", "Musk", "is", "the", "CEO", "of", "SpaceX", "."]
]
pos_tags = [
    ["NNP", "NNP", "VBD", "VBN", "IN", "NNP", "."],
    ["NNP", "NNP", "VBZ", "DT", "NN", "IN", "NNP", "."]
]
ner_tags = [
    ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O"],
    ["B-PER", "I-PER", "O", "O", "B-TITLE", "O", "B-ORG", "O"]
]

# Label mapping for POS and NER tags
unique_pos_tags = sorted(set(tag for tags in pos_tags for tag in tags))
unique_ner_tags = sorted(set(tag for tags in ner_tags for tag in tags))

pos_tag2id = {tag: i for i, tag in enumerate(unique_pos_tags)}
ner_tag2id = {tag: i for i, tag in enumerate(unique_ner_tags)}

# Convert to Dataset
data = {
    "tokens": tokens,
    "pos_tags": [[pos_tag2id[tag] for tag in tags] for tags in pos_tags],
    "ner_tags": [[ner_tag2id[tag] for tag in tags] for tags in ner_tags]
}
dataset = Dataset.from_dict(data)
dataset = DatasetDict({"train": dataset, "validation": dataset})  # Split into train and validation

# Tokenizer and model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(ner_tag2id))

# Tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load the metric using the new `evaluate` library
metric = load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[unique_ner_tags[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [unique_ner_tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  
    save_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,  
    metric_for_best_model="f1",
    logging_dir="./logs",
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# Save the model
trainer.save_model("ner-transformer")
tokenizer.save_pretrained("ner-transformer")
