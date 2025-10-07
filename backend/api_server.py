from flask import Flask, request, jsonify
from flask_cors import CORS
from fact_checker import DistilBertFactChecker

app = Flask(__name__)
CORS(app)

# Initialize DistilBERT-enhanced fact checker
print("=" * 60)
print("🤖 Loading DistilBERT-Enhanced Fact Checker...")
print("=" * 60)

fact_checker = DistilBertFactChecker(model_path="./model_output")

print("=" * 60)
print("✅ Fact Checker loaded successfully!")
print("📊 Strategy: DistilBERT analyzes claim + all site responses")
print("=" * 60)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "DistilBERT-Enhanced Fact Checker API is running",
        "model": "DistilBERT",
        "strategy": "Analyzes claim + site responses with DistilBERT"
    }), 200

@app.route('/check', methods=['POST'])
def check_fact():
    """
    Fact-check text using DistilBERT to analyze claim + site responses
    
    Request body:
    {
        "text": "The claim to fact-check"
    }
    
    Response:
    {
        "overall_verdict": "TRUE" | "FALSE",
        "overall_confidence": 0.85,
        "reasoning": "DistilBERT analysis explanation",
        "analysis_details": {
            "initial_prediction": "...",
            "site_responses_analyzed": 10,
            "combined_prediction": "..."
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request"
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        print(f"\n📝 Checking text: {text[:100]}...")
        print("=" * 60)
        
        # Perform DistilBERT-enhanced fact check
        result = fact_checker.check_text(text, verify_claims=True)
        
        # Format response for Android
        response = {
            "overall_verdict": result.get("overall_verdict", "UNCERTAIN"),
            "overall_confidence": result.get("overall_confidence", 0.0),
            "timestamp": result.get("timestamp"),
            "claims_count": len(result.get("claims", [])),
            "reasoning": ""
        }
        
        # Build reasoning from claims
        if result.get("claims"):
            first_claim = result["claims"][0]
            response["reasoning"] = first_claim.get("reasoning", "No reasoning available")
            
            # Add analysis details
            details = first_claim.get("details", {})
            response["analysis_details"] = {
                "initial_prediction": details.get("initial_analysis", {}).get("prediction"),
                "initial_confidence": details.get("initial_analysis", {}).get("confidence"),
                "site_responses_analyzed": details.get("site_analysis", {}).get("analyzed_count", 0),
                "web_results_found": details.get("web_results_count", 0),
                "factcheck_results_found": details.get("factcheck_results_count", 0),
                "combined_true_prob": details.get("combined_true_prob", 0),
                "combined_false_prob": details.get("combined_false_prob", 0)
            }
        else:
            response["reasoning"] = "Unable to extract verifiable claims"
            response["analysis_details"] = {}
        
        print(f"\n✅ Result: {response['overall_verdict']} ({response['overall_confidence']:.2%})")
        if "analysis_details" in response:
            print(f"   Analyzed {response['analysis_details'].get('site_responses_analyzed', 0)} site responses")
        print("=" * 60)
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": str(e),
            "overall_verdict": "ERROR",
            "overall_confidence": 0.0,
            "reasoning": f"An error occurred: {str(e)}"
        }), 500

@app.route('/verify-claim', methods=['POST'])
def verify_single_claim():
    """
    Verify a single claim with DistilBERT analyzing site responses
    
    Request body:
    {
        "claim": "Single factual claim to verify"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'claim' not in data:
            return jsonify({
                "error": "Missing 'claim' field in request"
            }), 400
        
        claim = data['claim']
        print(f"\n🔍 Verifying claim: {claim[:100]}...")
        
        # Verify single claim
        result = fact_checker.verify_claim(claim)
        
        details = result.get("details", {})
        
        response = {
            "verdict": result.get("verdict", "UNCERTAIN"),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "analysis_details": {
                "initial_prediction": details.get("initial_analysis", {}).get("prediction"),
                "initial_confidence": details.get("initial_analysis", {}).get("confidence"),
                "site_responses_analyzed": details.get("site_analysis", {}).get("analyzed_count", 0),
                "web_results_found": details.get("web_results_count", 0),
                "factcheck_results_found": details.get("factcheck_results_count", 0),
                "combined_true_prob": details.get("combined_true_prob", 0),
                "combined_false_prob": details.get("combined_false_prob", 0)
            }
        }
        
        print(f"✅ Verdict: {response['verdict']} ({response['confidence']:.2%})")
        print(f"   Site responses analyzed: {response['analysis_details']['site_responses_analyzed']}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            "error": str(e),
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🤖 DISTILBERT-ENHANCED FACT CHECKER API SERVER")
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /health          - Health check")
    print("  POST /check           - Fact-check text")
    print("  POST /verify-claim    - Verify single claim")
    print("=" * 60)
    print("\n💡 How it works:")
    print("  1. DistilBERT analyzes the original claim")
    print("  2. Searches web + fact-checking sites")
    print("  3. DistilBERT analyzes EACH site response")
    print("  4. Combines all predictions for final verdict")
    print("\n🌐 Starting server on http://0.0.0.0:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)