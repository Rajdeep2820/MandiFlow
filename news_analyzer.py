import json
import os
import re
from google import genai

class NewsAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            # New GenAI Client Initialization
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
        
        self.system_prompt = """
        You are an Indian agricultural commodity market intelligence AI.
        Analyze the following news or policy document chunk.
        Extract mathematical shock impacts for the supply chain using this EXACT JSON schema:

        {
            "commodities_affected": ["Onion", "Potato"],
            "origin_mandi": "LASALGAON (NIPHAD)",
            "origin_district": "Nashik",
            "shock_type": "climatic",
            "impact_multiplier": 1.5
        }

        Rules:
        1. origin_mandi must be the most specific physical mandi / auction yard / market named or implied by the text, not a generic state.
        2. If the text mentions a district or city but the agricultural mandi is commonly known by a related market yard, choose the likely mandi name.
        3. If it's a strike/blockade, use "logistics" and multiplier > 1.0.
        4. If it's heavy rain/crop damage, use "climatic" and multiplier > 1.0.
        5. If it's an export ban, use "policy" and multiplier < 1.0 for origin.
        6. If there is no clear location, set both origin_mandi and origin_district to "Unknown".
        7. Only respond with raw JSON. No markdown.
        """

    def _mock_llm_response(self, text):
        text_lower = text.lower()
        shock_type = "demand"
        if any(w in text_lower for w in ["rain", "flood", "drought", "weather", "hailstorm", "crop damage"]):
            shock_type = "climatic"
        elif any(w in text_lower for w in ["strike", "transport", "highway", "truck", "blockade", "jam"]):
            shock_type = "logistics"
        elif any(w in text_lower for w in ["ban", "policy", "tax", "export", "import", "duty"]):
            shock_type = "policy"

        multiplier = 1.0
        if shock_type in ["climatic", "logistics"]:
            multiplier = 1.3
        if "ban" in text_lower and "export" in text_lower:
            multiplier = 0.7

        mandi_aliases = [
            ("lasalgaon (niphad)", "LASALGAON (NIPHAD)", "Nashik"),
            ("lasalgaon (vinchur)", "LASALGAON (VINCHUR)", "Nashik"),
            ("lasalgaon", "LASALGAON", "Nashik"),
            ("niphad", "LASALGAON (NIPHAD)", "Nashik"),
            ("vinchur", "LASALGAON (VINCHUR)", "Nashik"),
            ("nashik", "NASIK", "Nashik"),
            ("nasik", "NASIK", "Nashik"),
            ("azadpur", "AZADPUR", "Delhi"),
            ("delhi", "AZADPUR", "Delhi"),
            ("bangalore", "BANGALORE", "Bengaluru Urban"),
            ("bengaluru", "BANGALORE", "Bengaluru Urban"),
            ("gurgaon", "GURGAON", "Gurugram"),
            ("gurugram", "GURGAON", "Gurugram"),
            ("mandsaur", "MANDSAUR", "Mandsaur"),
            ("indore", "INDORE", "Indore"),
            ("nagpur", "NAGPUR", "Nagpur"),
            ("pune", "PUNE", "Pune"),
            ("mumbai", "MUMBAI", "Mumbai"),
        ]

        origin_mandi = "Unknown"
        origin_district = "Unknown"
        for alias, mandi_name, district_name in mandi_aliases:
            if alias in text_lower:
                origin_mandi = mandi_name
                origin_district = district_name
                break

        if origin_mandi == "Unknown":
            title_case_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
            if title_case_matches:
                candidate = max(title_case_matches, key=len).strip()
                origin_mandi = candidate
                origin_district = candidate

        return {
            "commodities_affected": ["Onion"],
            "origin_mandi": origin_mandi,
            "origin_district": origin_district,
            "shock_type": shock_type,
            "impact_multiplier": multiplier
        }

    def extract_shock_features(self, combined_text):
        if not self.api_key or not self.client:
            return self._mock_llm_response(combined_text)
            
        try:
            prompt = f"{self.system_prompt}\n\nClient Input: {combined_text}"
            
            # New generate_content syntax
            response = self.client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"⚠️ LLM API Error: {e}. Falling back to heuristic model.")
            return self._mock_llm_response(combined_text)
