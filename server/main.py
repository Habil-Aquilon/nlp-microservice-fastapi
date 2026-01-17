"""
FastAPI сервер для NLP микросервиса
"""

import numpy as np
import nltk
from fastapi import FastAPI, Form, HTTPException
from sklearn.decomposition import TruncatedSVD
from typing import Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка данных NLTK
try:
    nltk.download([
        "punkt",
        "averaged_perceptron_tagger",
        "wordnet",
        "omw-1.4",
        "maxent_ne_chunker",
        "words"
    ])
except Exception as e:
    logger.warning(f"Ошибка загрузки NLTK данных: {e}")

app = FastAPI(
    title="NLP Microservice",
    description="Микросервис для обработки естественного языка с TF-IDF, LSA и NLTK функциями",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ---------- Загрузка корпуса ----------
def load_corpus(filename: str = "corpus_example.txt"):
    """
    Загружает корпус текстов из файла
    """
    try:
        with open(filename, encoding="utf-8") as f:
            docs = []
            for line in f:
                line = line.strip().lower()
                if line:
                    docs.append(line)
            logger.info(f"Загружено {len(docs)} документов из {filename}")
            return docs
    except FileNotFoundError:
        logger.warning(f"Файл {filename} не найден, используется пример корпуса")
        # Возвращаем пример корпуса
        return [
            "машинное обучение это интересно",
            "глубокое обучение это часть машинного обучения",
            "нейронные сети используются в глубоком обучении",
            "python популярный язык программирования",
            "fastapi современный веб фреймворк"
        ]

# Инициализация корпуса
documents = load_corpus("corpus_example.txt")

# ---------- Предобработка ----------
tokens = []
for doc in documents:
    tokens.append(doc.split())

vocabulary = []
for doc in tokens:
    for word in doc:
        if word not in vocabulary:
            vocabulary.append(word)

doc_count = len(tokens)
word_count = len(vocabulary)

# ---------- TF-IDF (NumPy) ----------
tf = np.zeros((doc_count, word_count))
df = np.zeros(word_count)

for i in range(doc_count):
    for j in range(word_count):
        word = vocabulary[j]
        tf[i][j] = tokens[i].count(word) / len(tokens[i])
        if word in tokens[i]:
            df[j] += 1

idf = np.log((doc_count + 1) / (df + 1)) + 1
tfidf_matrix = tf * idf

logger.info(f"Инициализирован корпус: {doc_count} документов, {word_count} уникальных слов")

# ---------- API Endpoints ----------

@app.get("/")
async def root():
    """
    Корневой эндпоинт
    """
    return {
        "service": "NLP Microservice",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "GET /": "Эта страница",
            "POST /tf-idf": "Получить TF-IDF матрицу",
            "GET /bag-of-words": "Bag-of-Words для текста",
            "POST /lsa": "Латентный семантический анализ",
            "POST /text_nltk/tokenize": "Токенизация",
            "POST /text_nltk/stem": "Стемминг (английский)",
            "POST /text_nltk/lemmatize": "Лемматизация (английский)",
            "POST /text_nltk/pos": "Частеречная разметка",
            "POST /text_nltk/ner": "Распознавание именованных сущностей"
        },
        "corpus_info": {
            "documents": doc_count,
            "vocabulary_size": word_count
        }
    }

@app.get("/health")
async def health_check():
    """
    Проверка здоровья сервиса
    """
    return {"status": "healthy"}

@app.post("/tf-idf")
async def get_tfidf():
    """
    Возвращает TF-IDF матрицу для корпуса
    """
    return {
        "matrix": tfidf_matrix.tolist(),
        "vocabulary": vocabulary,
        "documents": documents,
        "shape": {
            "rows": doc_count,
            "cols": word_count
        }
    }

@app.get("/bag-of-words")
async def bag_of_words(text: str):
    """
    Преобразует текст в Bag-of-Words вектор
    """
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    words = text.lower().split()
    vector = np.zeros(word_count, dtype=int)

    for i in range(word_count):
        if vocabulary[i] in words:
            vector[i] = 1

    return {
        "text": text,
        "vector": vector.tolist(),
        "vocabulary": vocabulary,
        "found_words": [w for w in words if w in vocabulary]
    }

@app.post("/lsa")
async def lsa_analysis(n_components: Optional[int] = 2):
    """
    Выполняет латентный семантический анализ
    """
    if n_components < 1 or n_components > min(tfidf_matrix.shape):
        raise HTTPException(
            status_code=400, 
            detail=f"n_components должен быть между 1 и {min(tfidf_matrix.shape)}"
        )
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    result = svd.fit_transform(tfidf_matrix)
    
    return {
        "matrix": result.tolist(),
        "explained_variance": svd.explained_variance_ratio_.tolist(),
        "total_variance": float(np.sum(svd.explained_variance_ratio_)),
        "components": n_components
    }

# ---------- NLTK Functions ----------

@app.post("/text_nltk/tokenize")
async def tokenize_text(text: str = Form(...)):
    """
    Токенизация текста
    """
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    tokens = nltk.word_tokenize(text)
    return {
        "text": text,
        "tokens": tokens,
        "count": len(tokens)
    }

@app.post("/text_nltk/stem")
async def stem_text(text: str = Form(...)):
    """
    Стемминг текста (английский)
    """
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    stemmer = nltk.stem.SnowballStemmer("english")
    words = nltk.word_tokenize(text)
    stems = [stemmer.stem(word) for word in words]
    
    return {
        "original": text,
        "tokens": words,
        "stems": stems
    }

@app.post("/text_nltk/lemmatize")
async def lemmatize_text(text: str = Form(...)):
    """
    Лемматизация текста (английский)
    """
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    
    return {
        "original": text,
        "tokens": words,
        "lemmas": lemmas
    }

@app.post("/text_nltk/pos")
async def pos_tagging(text: str = Form(...)):
    """
    Частеречная разметка (Part-of-Speech tagging)
    """
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    
    return {
        "text": text,
        "tokens": tokens,
        "pos_tags": tags
    }

@app.post("/text_nltk/ner")
async def named_entity_recognition(text: str = Form(...)):
    """
    Распознавание именованных сущностей (Named Entity Recognition)
    """
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(tags)
    
    entities = []
    for chunk in chunks:
        if hasattr(chunk, "label"):
            words = []
            for item in chunk:
                words.append(item[0])
            entities.append({
                "entity": " ".join(words),
                "label": chunk.label(),
                "type": get_entity_type(chunk.label())
            })
    
    return {
        "text": text,
        "tokens": tokens,
        "entities": entities,
        "count": len(entities)
    }

def get_entity_type(label: str) -> str:
    """
    Преобразует NLTK метки в читаемые типы
    """
    entity_types = {
        'PERSON': 'Person',
        'ORGANIZATION': 'Organization',
        'GPE': 'Geo-Political Entity',
        'LOCATION': 'Location',
        'DATE': 'Date',
        'TIME': 'Time',
        'MONEY': 'Money',
        'PERCENT': 'Percentage'
    }
    return entity_types.get(label, 'Other')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )