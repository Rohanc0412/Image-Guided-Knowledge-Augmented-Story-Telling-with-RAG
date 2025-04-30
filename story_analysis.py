import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from openai import OpenAI
import pandas as pd
import numpy as np
import torch
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import hashlib
from datetime import datetime
from nltk.tokenize import sent_tokenize
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY1")
client = OpenAI( api_key= OPENAI_API_KEY)

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model.eval()

SIMILARITY_THRESHOLD = 0.6
FACT_MATCH_THRESHOLD = 0.45

def story_coherence_score(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        coherence_sim = 0.0
    else:
        embeddings = semantic_model.encode(sentences)
        sims = [cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0][0] for i in range(len(embeddings)-1)]
        coherence_sim = np.mean(sims)

    try:
        readability_score = textstat.flesch_reading_ease(text)
        readability_scaled = min(readability_score / 100.0, 1.0)
    except:
        readability_score = 0.0
        readability_scaled = 0.0

    combined_score = round((0.7 * coherence_sim + 0.3 * readability_scaled), 3)
    return coherence_sim, readability_score, combined_score

def calculate_perplexity(text):
    encodings = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = gpt2_model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return round(perplexity.item(), 3)

def evaluate_story(caption, facts_text, story_text):
    story_vec = semantic_model.encode([story_text])[0].reshape(1, -1)
    caption_vec = semantic_model.encode([caption])[0].reshape(1, -1)
    summary_vec = semantic_model.encode([facts_text])[0].reshape(1, -1)
    return cosine_similarity(caption_vec, story_vec)[0][0], cosine_similarity(summary_vec, story_vec)[0][0]

def hallucination_rate(summary_sents, story_sents):
    fact_vecs = semantic_model.encode(summary_sents)
    story_vecs = semantic_model.encode(story_sents)
    hallucinated = sum(1 for sv in story_vecs if max(cosine_similarity(sv.reshape(1, -1), fv.reshape(1, -1))[0][0] for fv in fact_vecs) < 0.4)
    return round(hallucinated / len(story_sents), 3)

def fact_check_accuracy(summary_facts, story_sents, threshold=FACT_MATCH_THRESHOLD):
    summary_vecs = semantic_model.encode(summary_facts)
    story_vecs = semantic_model.encode(story_sents)
    scores = [max(cosine_similarity(sf.reshape(1, -1), sv.reshape(1, -1))[0][0] for sv in story_vecs) for sf in summary_vecs]
    supported = [(summary_facts[i], round(score, 3)) for i, score in enumerate(scores) if score >= threshold]
    unsupported = [(summary_facts[i], round(score, 3)) for i, score in enumerate(scores) if score < threshold]
    return round(len(supported) / len(summary_facts), 3), supported, unsupported

def run_story_analysis(caption, summary, story_text, save_dir="static/plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename hash for this story
    hash_id = hashlib.md5(story_text.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_prefix = f"story_{hash_id}_{timestamp}"

    story_sents = sent_tokenize(story_text)
    sum_sents = sent_tokenize(summary)

    caption_sim, summary_sim = evaluate_story(caption, summary, story_text)
    coherence, readability, combined = story_coherence_score(story_text)
    perplexity = calculate_perplexity(story_text)
    halluc_rate = hallucination_rate(sum_sents, story_sents)
    fact_acc, supported, unsupported = fact_check_accuracy(sum_sents, story_sents)

    metrics = pd.DataFrame([{
        "Caption_Story_Similarity": caption_sim,
        "Summary_Story_Similarity": summary_sim,
        "Semantic_Coherence": coherence,
        "Readability_Score": readability,
        "Combined_Coherence_Score": combined,
        "Perplexity": perplexity,
        "Hallucination_Rate": halluc_rate,
        "Fact_Check_Accuracy": fact_acc
    }])

    # Save overall heatmap
    heatmap1 = f"{filename_prefix}_overall.png"
    heatmap1_path = os.path.join(save_dir, heatmap1)
    plt.figure(figsize=(10, 4))
    sns.heatmap(metrics, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Story Evaluation Dashboard")
    plt.xticks(rotation=45)
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(heatmap1_path)
    plt.close()

    # Per-act metrics
    acts = re.split(r"\*\*Act [^\n]+\*\*", story_text)
    acts = [act.strip() for act in acts if act.strip()]
    acts = acts[1:]
    print("Split acts:", acts) 
    act_results = []

    for i, act_text in enumerate(acts):
        act_sents = sent_tokenize(act_text)
        caption_sim, summary_sim = evaluate_story(caption, summary, act_text)
        coherence, readability, combined = story_coherence_score(act_text)
        perplexity = calculate_perplexity(act_text)
        halluc_rate = hallucination_rate(sum_sents, act_sents)
        fact_acc, _, _ = fact_check_accuracy(sum_sents, act_sents)

        act_results.append({
            "Evaluation Metrics": f"Act {i+1}",
            "Caption_Story_Similarity": caption_sim,
            "Summary_Story_Similarity": summary_sim,
            "Semantic_Coherence": coherence,
            "Readability_Score": readability,
            "Combined_Coherence_Score": combined,
            "Perplexity": perplexity,
            "Hallucination_Rate": halluc_rate,
            "Fact_Check_Accuracy": fact_acc
        })

    act_df = pd.DataFrame(act_results)

    heatmap2 = f"{filename_prefix}_peract.png"
    heatmap2_path = os.path.join(save_dir, heatmap2)
    plt.figure(figsize=(12, 6))
    sns.heatmap(act_df.set_index("Evaluation Metrics"), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Per-Act Story Evaluation Dashboard")
    plt.ylabel("Acts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(heatmap2_path)
    plt.close()

    return metrics.T.rename(columns={0: "Score"}), act_df.set_index("Evaluation Metrics").T, supported, unsupported, heatmap1, heatmap2



def gpt_human_analysis(caption, summary, story_text):
    prompt = f"""
You are a professional fiction editor. Please evaluate the following story based on:
1. Relevance to the provided image caption
2. Factual consistency with the background summary
3. Coherence and structure across acts
4. Fluency and natural language use
5. Originality and creativity

Provide a brief qualitative critique and a score from 0–10 for each of the five criteria above.

---

Caption: {caption}

Summary: {summary}

Story: {story_text}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert story evaluator."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
