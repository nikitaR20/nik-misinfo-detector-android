import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def load_model(model_path="./model_output"):
    """Load the trained model and tokenizer"""
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict(text, model, tokenizer, device):
    """
    Predict if text is misinformation or not
    
    Args:
        text: The text to classify
        model: Trained model
        tokenizer: Tokenizer
        device: Device (CPU/GPU)
    
    Returns:
        dict: Prediction result with label and confidence
    """
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = "TRUE" if prediction == 1 else "FALSE"
    
    return {
        "text": text,
        "prediction": label,
        "confidence": confidence,
        "probabilities": {
            "FALSE": probabilities[0][0].item(),
            "TRUE": probabilities[0][1].item()
        }
    }

def main():
    print("=" * 60)
    print("🔍 Misinformation Detection - Testing")
    print("=" * 60)
    
    # Load model
    print("\n📂 Loading model...")
    try:
        model, tokenizer, device = load_model()
        print(f"✅ Model loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the model first using train.py")
        return
    
    # Test examples
    test_texts = [
        "Scientists have discovered a new planet in our solar system.",
        "Water is composed of hydrogen and oxygen.",
        "The Earth is flat and NASA is lying to us.",
        "Vaccines cause autism according to discredited studies.",
        "The moon landing happened in 1969."
    ]
    
    print("\n🧪 Testing sample texts:\n")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        result = predict(text, model, tokenizer, device)
        
        print(f"\n{i}. Text: {result['text'][:80]}...")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Probabilities:")
        print(f"     - FALSE (Misinformation): {result['probabilities']['FALSE']:.2%}")
        print(f"     - TRUE (Factual): {result['probabilities']['TRUE']:.2%}")
    
    print("\n" + "=" * 60)
    
    # Interactive mode
    print("\n🔄 Interactive Mode (type 'quit' to exit)\n")
    while True:
        user_input = input("Enter text to check: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = predict(user_input, model, tokenizer, device)
        print(f"\n📊 Result: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        print()

if __name__ == "__main__":
    main()