import torch
from transformers import AutoModel, AutoTokenizer

class BioInsightEmbedder:    
    def __init__(self, model_name, batch_size=32) -> None:
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_records(self, records: list) -> list:
        # 1. Verification: Ensure records actually have text
        texts = [getattr(r, 'text', '') for r in records]
        if not any(texts):
            print("Warning: No text found in records to embed.")
            return records

        embeddings = self._embed_texts(texts)
        
        # 2. Robust Update: Ensure the embedding is attached to the record object
        for i in range(len(records)):
            records[i].embedding = embeddings[i]
        
        return records
    
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # 3. Explicitly set max_length and truncation for BERT
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # BioBERT (BERT) standard is often Mean Pooling, 
            # but if you prefer CLS, ensure you detach correctly:
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
