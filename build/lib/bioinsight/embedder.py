import torch
from transformers import AutoTokenizer, AutoModel
from bioinsight.chroma_manager import BioInsightRecord

class BioInsightEmbedder:    
    def __init__(self, model, batch_size = 32) -> None:
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.model.eval()  # inference mode, not training
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def embed_records(self, records: list[BioInsightRecord]) -> list[BioInsightRecord]:
        texts = [record.text for record in records]
        embeddings = self._embed_texts(texts)
        
        for record, embedding in zip(records, embeddings):
            record.embedding = embedding
        
        return records
    
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embedding
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings