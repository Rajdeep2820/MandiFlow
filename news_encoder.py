import torch
from transformers import BertTokenizer, BertModel

class NewsEncoder:
    def __init__(self):
        # Load the pre-trained 'Brain' for language (BERT)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        print("📰 BERT News Encoder Loaded.")

    def get_shock_vector(self, headline):
        # 1. Tokenization: Break sentence into math-ready pieces
        inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True)
        
        # 2. Forward Pass through BERT
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. Math: Take the 'CLS' token (the summary vector of the whole sentence)
        # This vector represents the "meaning" of the news
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

# Test it
encoder = NewsEncoder()
vector = encoder.get_shock_vector("Heavy rainfall floods Mandsaur Mandi garlic stocks")
print(f"News Vector Shape: {vector.shape}") # Should be [1, 768]