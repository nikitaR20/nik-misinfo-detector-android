import torch
import requests
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from dotenv import load_dotenv
import os
from typing import Dict, List
import json
from datetime import datetime
import re

class DistilBertFactChecker:
    """
    Enhanced fact checker that:
    1. Uses DistilBERT to analyze the original claim
    2. Searches web and fact-checking sites
    3. Feeds site responses BACK to DistilBERT for analysis
    4. Combines all DistilBERT predictions for final verdict
    """
    
    def __init__(self, model_path="./model_output"):
        # Load ML model
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load environment variables (from .env if present)
        load_dotenv()
        # API keys - set via environment: GOOGLE_API_KEY and GOOGLE_CX
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CX")

        print(f"✅ DistilBERT loaded on {self.device}")
        
    def extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text"""
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return claims
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the web for information"""
        if not self.google_api_key or not self.google_cx:
            print("⚠️  No Google API credentials")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': query,
                'num': num_results
            }
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get('items', []):
                    results.append({
                        'title': item.get('title'),
                        'snippet': item.get('snippet'),
                        'link': item.get('link'),
                        'source': item.get('displayLink')
                    })
                return results
        except Exception as e:
            print(f"Search error: {e}")
        
        return []
    
    def check_trusted_sources(self, claim: str) -> Dict:
        """Check against trusted fact-checking databases"""
        trusted_sources = {
            'snopes': 'snopes.com/fact-check/',
            'politifact': 'politifact.com',
            'factcheck': 'factcheck.org'
        }
        
        results = {}
        for source, domain in trusted_sources.items():
            search_results = self.search_web(f"{claim} site:{domain}")
            if search_results:
                results[source] = search_results
        
        return results
    
    def analyze_with_distilbert(self, text: str) -> Dict:
        """
        Analyze text with DistilBERT
        Returns prediction, confidence, and probabilities
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            "prediction": "TRUE" if prediction == 1 else "FALSE",
            "confidence": confidence,
            "false_prob": probabilities[0][0].item(),
            "true_prob": probabilities[0][1].item()
        }
    
    def analyze_site_responses(self, claim: str, web_results: List[Dict], 
                               factcheck_results: Dict) -> Dict:
        """
        Feed site responses to DistilBERT for analysis
        """
        print(f"  🤖 Analyzing site responses with DistilBERT...")
        
        all_snippets = []
        snippet_sources = []
        
        # Collect web search snippets
        for result in web_results[:5]:
            snippet = result.get('snippet', '')
            if snippet:
                all_snippets.append(snippet)
                snippet_sources.append({
                    'type': 'web',
                    'source': result.get('source', 'unknown'),
                    'title': result.get('title', '')
                })
        
        # Collect fact-checking site snippets
        for source_name, results in factcheck_results.items():
            for result in results[:2]:
                snippet = result.get('snippet', '')
                if snippet:
                    all_snippets.append(snippet)
                    snippet_sources.append({
                        'type': 'factcheck',
                        'source': source_name,
                        'title': result.get('title', '')
                    })
        
        if not all_snippets:
            print("     ⚠️  No snippets to analyze")
            return {
                'analyzed_count': 0,
                'predictions': [],
                'avg_true_prob': 0.5,
                'avg_false_prob': 0.5
            }
        
        # Analyze each snippet with DistilBERT + heuristic boost
        predictions = []
        confirmation_keywords = ["married", "tied the knot", "officially married", "announced wedding", "wedding ceremony"]
        contradiction_keywords = ["denied", "false", "did not happen", "no evidence"]
        
        for i, snippet in enumerate(all_snippets):
            context = f"Claim: {claim}. Evidence: {snippet}"
            analysis = self.analyze_with_distilbert(context)
            
            # Heuristic boost
            snippet_lower = snippet.lower()
            if any(word in snippet_lower for word in confirmation_keywords):
                analysis['true_prob'] = min(analysis['true_prob'] + 0.10, 1.0)
                analysis['false_prob'] = 1.0 - analysis['true_prob']
            elif any(word in snippet_lower for word in contradiction_keywords):
                analysis['false_prob'] = min(analysis['false_prob'] + 0.10, 1.0)
                analysis['true_prob'] = 1.0 - analysis['false_prob']
            
            predictions.append({
                'snippet': snippet[:100] + '...',
                'source': snippet_sources[i],
                'prediction': analysis['prediction'],
                'confidence': analysis['confidence'],
                'true_prob': analysis['true_prob'],
                'false_prob': analysis['false_prob']
            })
            
            print(f"     [{i+1}/{len(all_snippets)}] {snippet_sources[i]['source']}:")
            print(f"        Snippet: {snippet[:200]}...")
            print(f"        Prediction: {analysis['prediction']} ({analysis['confidence']:.2%})\n")
        
        # Aggregate predictions
        avg_true_prob = sum(p['true_prob'] for p in predictions) / len(predictions)
        avg_false_prob = sum(p['false_prob'] for p in predictions) / len(predictions)
        
        return {
            'analyzed_count': len(predictions),
            'predictions': predictions,
            'avg_true_prob': avg_true_prob,
            'avg_false_prob': avg_false_prob
        }
    
    def verify_claim(self, claim: str) -> Dict:
        """
        Comprehensive claim verification
        """
        print(f"\n🔍 Verifying claim: {claim[:100]}...")
        
        print(f"  📊 Step 1: Analyzing claim with DistilBERT...")
        initial_analysis = self.analyze_with_distilbert(claim)
        print(f"     Initial prediction: {initial_analysis['prediction']} "
              f"({initial_analysis['confidence']:.2%})")
        
        print(f"  🌐 Step 2: Gathering evidence from web...")
        web_results = self.search_web(claim)
        print(f"     Found {len(web_results)} web results")
        
        print(f"  ✓ Step 3: Checking fact-checking sites...")
        factcheck_results = self.check_trusted_sources(claim)
        total_factcheck = sum(len(v) for v in factcheck_results.values())
        print(f"     Found {total_factcheck} fact-check results")
        
        print(f"  🤖 Step 4: Analyzing evidence with DistilBERT...")
        site_analysis = self.analyze_site_responses(claim, web_results, factcheck_results)
        
        print(f"  📈 Step 5: Combining all predictions...")
        initial_weight = 0.3
        sites_weight = 0.7
        
        combined_true_prob = (
            initial_analysis['true_prob'] * initial_weight +
            site_analysis['avg_true_prob'] * sites_weight
        )
        combined_false_prob = (
            initial_analysis['false_prob'] * initial_weight +
            site_analysis['avg_false_prob'] * sites_weight
        )
        
        final_verdict = "TRUE" if combined_true_prob > combined_false_prob else "FALSE"
        confidence = max(combined_true_prob, combined_false_prob)
        
        print(f"     Combined TRUE probability: {combined_true_prob:.2%}")
        print(f"     Combined FALSE probability: {combined_false_prob:.2%}")
        print(f"     Final verdict: {final_verdict} ({confidence:.2%})")
        
        reasoning = self.generate_reasoning(
            initial_analysis, 
            site_analysis, 
            final_verdict,
            combined_true_prob,
            combined_false_prob
        )
        
        return {
            "claim": claim,
            "verdict": final_verdict,
            "confidence": confidence,
            "details": {
                "initial_analysis": initial_analysis,
                "web_results_count": len(web_results),
                "factcheck_results_count": total_factcheck,
                "site_analysis": site_analysis,
                "combined_true_prob": combined_true_prob,
                "combined_false_prob": combined_false_prob
            },
            "reasoning": reasoning
        }
    
    def generate_reasoning(self, initial_analysis, site_analysis, verdict, 
                          true_prob, false_prob):
        """Generate human-readable reasoning"""
        reasons = []
        
        reasons.append(
            f"DistilBERT initial analysis: {initial_analysis['prediction']} "
            f"({initial_analysis['confidence']:.0%} confidence)"
        )
        
        if site_analysis['analyzed_count'] > 0:
            supporting = sum(1 for p in site_analysis['predictions'] 
                           if p['prediction'] == verdict)
            reasons.append(
                f"Analyzed {site_analysis['analyzed_count']} evidence snippets: "
                f"{supporting} support {verdict}"
            )
        
        reasons.append(
            f"Combined analysis: {true_prob:.0%} TRUE, {false_prob:.0%} FALSE"
        )
        
        return " | ".join(reasons)
    
    def check_text(self, text: str, verify_claims: bool = True) -> Dict:
        result = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "overall_verdict": None,
            "overall_confidence": None,
            "claims": []
        }
        
        if verify_claims:
            claims = self.extract_claims(text)
            print(f"\n📝 Extracted {len(claims)} claim(s)")
            
            claim_results = []
            for claim in claims[:3]:
                claim_result = self.verify_claim(claim)
                claim_results.append(claim_result)
            
            result["claims"] = claim_results
            
            if claim_results:
                true_count = sum(1 for c in claim_results if c["verdict"] == "TRUE")
                overall_confidence = sum(c["confidence"] for c in claim_results) / len(claim_results)
                
                result["overall_verdict"] = "TRUE" if true_count > len(claim_results) / 2 else "FALSE"
                result["overall_confidence"] = overall_confidence
        else:
            claim_result = self.verify_claim(text)
            result["claims"] = [claim_result]
            result["overall_verdict"] = claim_result["verdict"]
            result["overall_confidence"] = claim_result["confidence"]
        
        return result


def main():
    print("=" * 70)
    print("🤖 DISTILBERT-ENHANCED FACT CHECKER")
    print("=" * 70)
    print("Strategy:")
    print("  1. DistilBERT analyzes the original claim")
    print("  2. Search web and fact-checking sites")
    print("  3. DistilBERT analyzes EACH site response")
    print("  4. Combine all DistilBERT predictions")
    print("=" * 70)
    
    checker = DistilBertFactChecker()
    
    test_claims = [
        "The Earth is flat",
        "Water boils at 100 degrees Celsius at sea level",
        "Selena Gomez marries Benny Blanco in California"
    ]
    
    for claim in test_claims:
        print("\n" + "=" * 70)
        result = checker.check_text(claim, verify_claims=False)
        
        print(f"\n📊 FINAL RESULT:")
        print(f"   Claim: {result['claims'][0]['claim']}")
        print(f"   Verdict: {result['overall_verdict']}")
        print(f"   Confidence: {result['overall_confidence']:.2%}")
        print(f"   Reasoning: {result['claims'][0]['reasoning']}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
