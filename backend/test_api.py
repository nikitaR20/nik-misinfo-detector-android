import requests

BASE_URL = "http://127.0.0.1:5000"  # FastAPI server URL
TEXT = "Selena Gomez marries Benny Blanco in California"  # sample query

def test_check_fact():
    url = f"{BASE_URL}/check"
    payload = {"text": TEXT}

    print(f"🔍 Sending request to {url} with text: '{TEXT}'...")
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"❌ API call failed: {response.status_code} {response.text}")
        return

    data = response.json()
    print("✅ API Response received\n")

    # Check local model result
    if "local_model" in data:
        print("📌 Local Model Prediction:")
        print(data["local_model"])
    else:
        print("⚠️ No local model result found")

    # Check Google API result
    if "google_api" in data:
        print("\n🌍 Google Fact Check API Results:")
        if isinstance(data["google_api"], list) and len(data["google_api"]) > 0:
            for claim in data["google_api"][:3]:  # print top 3
                print(f"- Claim: {claim.get('text')}")
                if "claimReview" in claim:
                    for review in claim["claimReview"]:
                        print(f"  - Source: {review['publisher'].get('name')}")
                        print(f"  - Rating: {review.get('textualRating')}")
        else:
            print("⚠️ No claims returned from Google API")
    else:
        print("⚠️ Google API field missing in response")

if __name__ == "__main__":
    test_check_fact()
