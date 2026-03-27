import json
import os
import google.generativeai as genai

class NewsAnalyzer:
    def __init__(self, api_key=None):
        """
        Decoupled feature extraction layer to turn unstructured news/docs into mathematical shocks.
        Uses Google Gemini for zero-shot JSON extraction.
        """
        self.api_key = api_key or os.getenv("AIzaSyAVCB4Ps7W-ns9TQ7XNxrwEYRvaNFau8q8", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
        
        # System Prompt Engineering
        self.system_prompt = """
        You are an Indian agricultural commodity market intelligence AI. 
        Analyze the following news or policy document chunk. 
        Extract mathematical shock impacts for the supply chain using this EXACT JSON schema:
        
        {
            "commodities_affected": ["Onion", "Potato"], 
            "origin_mandi": "Mandsaur", 
            "shock_type": "climatic", // Must be: 'climatic', 'logistics', 'policy', 'demand'
            "impact_multiplier": 1.5 // e.g. 1.5 = 50% spike, 0.7 = 30% drop
        }
        
        Rules:
        1. If it's a strike/blockade, use 'logistics' and multiplier > 1.0.
        2. If it's heavy rain/crop damage, use 'climatic' and multiplier > 1.0.
        3. If it's an export ban, use 'policy' and multiplier < 1.0 for origin.
        4. Only respond with raw JSON. No markdown.
        """

    def _mock_llm_response(self, text):
        """Fallback deterministic parsing if no API configured or if API fails."""
        text_lower = text.lower()
        shock_type = "demand"
        if any(w in text_lower for w in ["rain", "flood", "drought", "weather"]): shock_type = "climatic"
        elif any(w in text_lower for w in ["strike", "transport", "highway", "truck"]): shock_type = "logistics"
        elif any(w in text_lower for w in ["ban", "policy", "tax", "export", "import"]): shock_type = "policy"
            
        multiplier = 1.0
        if shock_type in ["climatic", "logistics"]: multiplier = 1.3
        if "ban" in text_lower and "export" in text_lower: multiplier = 0.7
            
        origin = "Unknown"
        for city in ["Mandsaur", "Nashik", "Azadpur", "Mumbai", "Pune", "Indore", "Nagpur"]:
            if city.lower() in text_lower:
                origin = city
                break
        
        comms = []
        for c in ["Onion", "Potato", "Tomato", "Wheat", "Rice"]:
            if c.lower() in text_lower: comms.append(c)
        if not comms: comms.append("All")
        
        return {
            "commodities_affected": comms,
            "origin_mandi": origin,
            "shock_type": shock_type,
            "impact_multiplier": multiplier
        }

    def extract_shock_features(self, combined_text):
        """Takes raw text and outputs structured JSON via Gemini or Mock."""
        if not self.api_key or not self.model:
            return self._mock_llm_response(combined_text)
            
        try:
            prompt = f"{self.system_prompt}\n\nClient Input: {combined_text}"
            response = self.model.generate_content(prompt)
            
            # Clean potential markdown backticks
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"⚠️ LLM API Error: {e}. Falling back to heuristic model.")
            return self._mock_llm_response(combined_text)

if __name__ == "__main__":
    # Test it directly
    analyzer = NewsAnalyzer()
    print("Test Result:", analyzer.extract_shock_features("Heavy rain in Nashik destroyed onion crops."))