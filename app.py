import os
from flask import Flask, request, render_template, redirect, session, url_for, send_from_directory
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import markdown
import uuid

load_dotenv()

from image_caption_generation import generate_best_caption
from information_retrieval import retrieve_facts_from_vector_db, get_facts_from_google
from story_generation import query_model
from story_analysis import run_story_analysis, gpt_human_analysis
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = "super_secret_dev_key_123"
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

STORY_DIR = "generated_stories"
os.makedirs(STORY_DIR, exist_ok=True)

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION = os.environ.get("PINECONE_REGION")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "llm-proj"
idx = pc.Index(index_name)

embedding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

@app.route('/', methods=['GET', 'POST'])
def index():
    print("in index")
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('result', filename=file.filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result/<filename>')
def result(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(image_path).convert("RGB")
    
    best_caption = generate_best_caption(image)
    
    facts = retrieve_facts_from_vector_db(best_caption, embedding_model, idx, threshold=0.5, top_k=5)
    if facts:
        print("Retrieved from Vector DB")
        article_summary = facts[0].get('event_text', str(facts[0]))
    else:
        print("Retrieved from google")
        article_summary = get_facts_from_google(best_caption, SERPAPI_API_KEY)[0]
    
    generated_story = query_model(best_caption, article_summary)

    story_id = str(uuid.uuid4())
    story_path = os.path.join(STORY_DIR, f"{story_id}.txt")

    with open(story_path, "w", encoding="utf-8") as f:
        f.write(generated_story)

    session["story_id"] = story_id
    session["best_caption"] = best_caption
    session["summary"] = article_summary

    
    generated_story_html = markdown.markdown(generated_story, extensions=["extra", "codehilite", "toc"])
    
    image_url = url_for('uploaded_file', filename=filename)
    return render_template('result.html', caption=best_caption, summary=article_summary, story=generated_story_html, image_url=image_url)

@app.route('/analysis')
def analysis():
    story_id = session.get("story_id")
    if not story_id:
        return redirect(url_for("index"))

    story_path = os.path.join(STORY_DIR, f"{story_id}.txt")

    # Read it dynamically
    with open(story_path, "r", encoding="utf-8") as f:
        story = f.read()

    caption = session.get("best_caption")
    summary = session.get("summary")

    if not story or not caption or not summary:
        return redirect(url_for('index'))
    print("Printing story")
    print(story)
    print("Printing caption")
    print(caption)
    print("Printing summary")
    print(summary)
    overall, per_act, supported, unsupported, heatmap1, heatmap2 = run_story_analysis(caption, summary, story)
    gpt_review = gpt_human_analysis(caption, summary, story)

    return render_template("analysis.html",
                           overall=overall.to_html(classes="table table-striped",float_format="{:.3f}".format),
                           per_act=per_act.to_html(classes="table table-striped table-hover table-sm align-middle text-start",float_format="{:.3f}".format,index=True),
                           supported=supported,
                           unsupported=unsupported,
                           gpt_review=gpt_review,
                           heatmap_url=url_for('static', filename=f'plots/{heatmap1}'),
                           per_act_heatmap_url=url_for('static', filename=f'plots/{heatmap2}'))


if __name__ == '__main__':
    app.run(port=82, debug=True)