import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import os

from dataset import load_all_datasets
from preprocess import preprocess_dataframe, MisinformationDataset

def main():
    print("=" * 50)
    print("🚀 Starting Misinformation Detection Training")
    print("=" * 50)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    datasets = load_all_datasets()
    
    if not datasets:
        print("❌ No datasets found! Please check your dataset directory.")
        return
    
    print(f"\n✅ Datasets found: {list(datasets.keys())}")

    # Select dataset (prioritize train, then first available)
    if "train" in datasets:
        df = datasets["train"]
        dataset_name = "train"
    else:
        dataset_name = list(datasets.keys())[0]
        df = datasets[dataset_name]

    print(f"\n📊 Using dataset: '{dataset_name}' with {len(df)} samples")

    # Preprocess
    print("\n🔄 Preprocessing data...")
    encodings, labels = preprocess_dataframe(df)
    
    if not labels:
        print("❌ No valid samples after preprocessing!")
        return
    
    print(f"✅ Preprocessed {len(labels)} samples")
    print(f"Label distribution: {dict(pd.Series(labels).value_counts())}")

    # Create Dataset and DataLoader
    dataset = MisinformationDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"✅ Created DataLoader with {len(dataloader)} batches")

    # Load model
    print("\n🤖 Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.to(device)
    model.train()
    print("✅ Model loaded successfully")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    print("\n🎯 Starting training...")
    print("-" * 50)
    
    num_epochs = 3  # Change this to train for more epochs
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )

            loss = outputs.loss
            logits = outputs.logits

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100 * correct / total
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"\n✅ Epoch {epoch + 1} Complete!")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.2f}%")

    # Save model
    print("\n💾 Saving model...")
    output_dir = "./model_output"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    print("\n" + "=" * 50)
    print("🎉 Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()