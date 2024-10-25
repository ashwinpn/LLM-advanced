
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from transformers import pipeline
import plotly.graph_objects as go

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to compute similarity between resume and job description
def compute_similarity(resume_text, job_description):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    
    # Calculate the cosine similarity
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding)
    
    return similarity_score.item()  # Returns a similarity score between 0 and 1

# Function to compute keyword-based matching using TF-IDF
def keyword_matching(resume_text, job_description):
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity between the two
    cosine_sim = (tfidf_matrix[0] * tfidf_matrix[1].T).A[0][0]
    return cosine_sim  # Returns a cosine similarity score

# Initialize spaCy for extracting skills/entities
nlp = spacy.load("en_core_web_sm")

# Function to extract skills using spaCy NER
def extract_spacy_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "EXPERIENCE", "ORG", "TECH", "LANGUAGE"]]
    return set(skills)

# Load a pre-trained Hugging Face GPT-2 model for text generation
generator = pipeline('text-generation', model='gpt2')

# Use LLMs to fill in gaps if spaCy misses some skills or entities
def extract_llm_entities(text):
    prompt = f"Extract technical skills and relevant entities from the following text: {text}"
    
    # Call GPT-2 for entity extraction (this serves as a fallback if spaCy misses)
    response = generator(prompt, max_length=200, num_return_sequences=1)
    
    return response[0]['generated_text']

# Function to match skills between resume and job description using both spaCy and LLM
def match_skills(resume_text, job_description):
    resume_skills_spacy = extract_spacy_skills(resume_text)
    job_skills_spacy = extract_spacy_skills(job_description)
    
    # Combine spaCy extraction with LLM extraction
    resume_skills_llm = extract_llm_entities(resume_text)
    job_skills_llm = extract_llm_entities(job_description)
    
    # Combine results from spaCy and LLMs
    resume_skills = resume_skills_spacy.union(set(resume_skills_llm.split()))
    job_skills = job_skills_spacy.union(set(job_skills_llm.split()))
    
    # Find matched and missing skills
    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills
    
    return matched_skills, missing_skills

# Function to suggest improvements for missing skills
def suggest_improvements(missing_skills, job_description):
    prompt = f"The job description requires the following missing skills: {', '.join(missing_skills)}. Suggest ways to improve the resume."
    
    # Generate text suggestions based on the prompt
    suggestions = generator(prompt, max_length=200, num_return_sequences=1)
    
    return suggestions[0]['generated_text']

# Function to highlight matches in the PDF
def highlight_matches_in_pdf(resume_text, matched_skills):
    # Simple visual highlighting using Plotly (or other PDF libraries for annotations)
    words = resume_text.split()
    
    highlighted_text = ['<b>' + word + '</b>' if word in matched_skills else word for word in words]
    highlighted_resume = ' '.join(highlighted_text)
    
    return highlighted_resume

# Function to generate a report
def generate_report(resume_text, job_description):
    match_score = compute_similarity(resume_text, job_description)
    keyword_score = keyword_matching(resume_text, job_description)
    matched_skills, missing_skills = match_skills(resume_text, job_description)
    improvement_suggestions = suggest_improvements(missing_skills, job_description)
    
    report = {
        "Semantic Match Score": match_score,
        "Keyword Match Score": keyword_score,
        "Matched Skills": list(matched_skills),
        "Missing Skills": list(missing_skills),
        "Improvement Suggestions": improvement_suggestions
    }
    
    return report
