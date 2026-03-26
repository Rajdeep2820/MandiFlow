import json
import os
import requests

class NewsAnalyzer:
    def __init__(self, api_key=None):
        """
        Decoupled feature extraction layer to turn unstructured news/docs into mathematical shocks.
        Uses an LLM (mocked or via actual API).
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        
        # System Prompt Engineering
        self.system_prompt = """
        You are a commodity market intelligence AI. Analye the following news or policy document chunk.
        Extract the mathematical shock impacts for the agricultural supply chain using this EXACT JSON schema:
        
        {
            "commodities_affected": ["Onion", "Potato"], // List of commodities
            "origin_mandi": "Mandsaur", // The primary market or region affected (if any)
            "shock_type": "climatic", // Must be one of: 'climatic', 'logistics', 'policy', 'demand'
            "impact_multiplier": 1.5 // Multiplier on standard price (e.g. 1.5 means 50% price spike, 0.8 means 20% drop)
        }
        
        Only respond with the raw JSON object. Do not include markdown formatting or explanations.
        """

    def _mock_llm_response(self, text):
        """
        Fallback deterministic parsing if no API configured. 
        """
        text_lower = text.lower()
        
        # Determine Shock Type
        shock_type = "demand"
        if "rain" in text_lower or "flood" in text_lower or "drought" in text_lower:
            shock_type = "climatic"
        elif "strike" in text_lower or "transport" in text_lower or "highway" in text_lower:
            shock_type = "logistics"
        elif "ban" in text_lower or "policy" in text_lower or "tax" in text_lower:
            shock_type = "policy"
            
        # Determine Multiplier
        multiplier = 1.0
        if shock_type in ["climatic", "logistics"]:
            multiplier = 1.3 # Shortage implies price spike
        if "ban" in text_lower and "export" in text_lower:
            multiplier = 0.7 # Export ban crashes local origin price
            
        # Determine Origin
        origin = "Unknown"
        if "mandsaur" in text_lower: origin = "Mandsaur"
        elif "nashik" in text_lower: origin = "Nashik"
        elif "delhi" in text_lower: origin = "Azadpur (Delhi)"
        elif "mumbai" in text_lower: origin = "Mumbai"
        
        # Determine Commodities
        comms = []
        if "onion" in text_lower: comms.append("Onion")
        if "potato" in text_lower: comms.append("Potato")
        if "tomato" in text_lower: comms.append("Tomato")
        if not comms: comms.append("All")
        
        return {
            "commodities_affected": comms,
            "origin_mandi": origin,
            "shock_type": shock_type,
            "impact_multiplier": multiplier
        }

    def extract_shock_features(self, combined_text):
        """
        Takes raw text (news or PDF chunk) and outputs a structured JSON multiplier.
        """
        if not self.api_key:
            return self._mock_llm_response(combined_text)
            
        # Example of a real LLM call (e.g. OpenAI or Gemini). 
        # Here we mock the request structure to show how it plugs in.
        try:
            # Simulated API Call
            payload = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"News/Doc: {combined_text}"}
                ]
            }
            # response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.api_key}"})
            # content = response.json()['choices'][0]['message']['content']
            # return json.loads(content)
            
            # Since we didn't actually hit an API to avoid billing/keys issue here, we use mock
            return self._mock_llm_response(combined_text)
        except Exception as e:
            print(f"Error parsing LLM JSON: {e}")
            return self._mock_llm_response(combined_text)

# Test It
if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    test_news = "Truckers strike on Delhi-Mumbai highway halts onion transport."
    print("Test Output:", json.dumps(analyzer.extract_shock_features(test_news), indent=2))
