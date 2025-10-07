from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Step 1: Load pre-trained tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 2: Test with example text
texts = [
    "COVID-19 can be cured by drinking bleach.",
    "NASA landed astronauts on the moon in 1969."
]

# Step 3: Tokenize
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Step 4: Run model
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

# Step 5: Print results
for text, prob in zip(texts, probs):
    print(f"Text: {text}")
    print(f"Probability [True, False]: {prob.tolist()}")
    print("-"*50)
