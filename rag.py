from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAG:
    def __init__(self):
        self.m3e = SentenceTransformer("/home/NLP/CMQA-LIA/m3e-small")
        self.knowledge_base = [
            "感冒常用药物：布洛芬、对乙酰氨基酚，注意过敏禁忌",
            "糖尿病典型症状：多饮、多食、多尿、体重减轻",
            "高血压需低盐饮食，定期监测血压",
            "肺炎常见症状：发热、咳嗽、咳痰、胸闷气短",
            "阿莫西林需在医生指导下使用，避免过敏风险"
        ]
        self.embeds = self.m3e.encode(self.knowledge_base, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeds.shape[1])
        self.index.add(self.embeds)

    def retrieve(self, query, top_k=3):
        q_emb = self.m3e.encode([query], convert_to_numpy=True)
        _, idx = self.index.search(q_emb, top_k)
        return [self.knowledge_base[i] for i in idx[0]]