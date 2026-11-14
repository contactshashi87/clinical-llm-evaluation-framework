import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .paths import ROOT

KB = ROOT / 'data' / 'shared' / 'knowledge_base.md'

def load_chunks():
    text = KB.read_text(encoding='utf-8')
    chunks = [c.strip() for c in re.split(r'\n## ', text) if c.strip()]
    return chunks

class LocalRetriever:
    def __init__(self, top_k=4):
        self.chunks = load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.top_k = top_k
    def retrieve(self, query):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = scores.argsort()[::-1][:self.top_k]
        return [self.chunks[i] for i in idx if scores[i] > 0] or self.chunks[:1]
