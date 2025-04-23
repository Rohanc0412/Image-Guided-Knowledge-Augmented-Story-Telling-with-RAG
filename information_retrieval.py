import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

unwanted_keywords = [
    "Shop by", "Men's", "Women's", "Tops", "Bottoms",
    "Sign up", "Subscribe", "Log in", "Create Account",
    "Customer Service", "My Orders", "Track Order"
]
unwanted_pattern = re.compile("|".join(map(re.escape, unwanted_keywords)), re.IGNORECASE)

def cosine_sim(a, b):
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def retrieve_facts_from_vector_db(caption, embedding_model, index, threshold=0.3, top_k=5):
    try:
        caption_embedding = embedding_model.encode([caption])[0]
        caption_embedding_np = np.array(caption_embedding, dtype=np.float32)
        query_response = index.query(
            vector=caption_embedding_np.tolist(),
            top_k=top_k,
            include_values=True,
            include_metadata=True,
        )
        matches = query_response.get("matches", [])
        if not matches:
            return None
        filtered_matches = []
        for match in matches:
            db_vector = np.array(match["values"], dtype=np.float32)
            sim_score = cosine_sim(caption_embedding_np, db_vector)
            print("sim_score:", sim_score)
            if sim_score >= threshold:
                filtered_matches.append(match["metadata"])
        return filtered_matches if filtered_matches else None
    except Exception as e:
        print("Vector DB Error:", e)
        return None

def extract_article_text(url, min_sentences=15):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        main_content = None
        for tag in ['article', 'main', 'div', 'section']:
            main_content = soup.find(tag)
            if main_content:
                break
        paragraphs = main_content.find_all("p") if main_content else soup.find_all("p")
        clean_paragraphs = [
            p.get_text().strip() for p in paragraphs
            if len(p.get_text().strip()) > 40 and not unwanted_pattern.search(p.get_text())
        ]
        sentences = re.split(r'(?<=[.!?]) +', " ".join(clean_paragraphs))
        sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
        return " ".join(sentences[:min_sentences]) if len(sentences) >= min_sentences else None
    except Exception:
        return None

def get_facts_from_google(caption, serpapi_api_key, max_articles=3):
    search_url = "https://serpapi.com/search"
    params = {
        "q": f"real life event involving {caption.lower()}",
        "hl": "en",
        "gl": "us",
        "api_key": serpapi_api_key
    }
    try:
        response = requests.get(search_url, params=params)
        results = response.json()
        if "organic_results" not in results:
            return ["No search results found."]
        articles = []
        for result in results["organic_results"]:
            url = result.get("link")
            if url:
                article = extract_article_text(url)
                if article:
                    articles.append(article)
                if len(articles) >= max_articles:
                    break
        return articles if articles else ["Could not extract meaningful articles."]
    except Exception as e:
        print("Google Search Error:", e)
        return ["Google search failed."]
