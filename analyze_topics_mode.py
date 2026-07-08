import sqlite3
import os
import sys
import argparse
import json
import urllib.request
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def analyze_posts_ollama(posts_data, topics, lang, max_workers=8):
    results = []
    url = "http://localhost:11434/api/generate"
    topics_str = ", ".join(topics)
    
    # 1. Отсеиваем пустые и слишком короткие тексты сразу, чтобы не нагружать Ollama
    valid_posts = []
    for post_id, text in posts_data:
        text_str = str(text) if text is not None else ""
        if not text_str or len(text_str.strip()) < 10:
            results.append({
                "post_id": post_id,
                "topic": "Без категории" if lang == "ru" else "Uncategorized",
                "score": 0.0
            })
        else:
            valid_posts.append((post_id, text_str[:2000]))

    # 2. Функция для отправки одного HTTP-запроса в Ollama
    def process_single_post(item):
        post_id, text_str = item
        
        if lang == "ru":
            prompt = f"Классифицируй следующий текст строго в одну из этих категорий: {topics_str}.\n\nТекст: {text_str}\n\nОтветь ТОЛЬКО названием категории."
        else:
            prompt = f"Classify the following text strictly into one of these categories: {topics_str}.\n\nText: {text_str}\n\nAnswer ONLY with the category name."
            
        payload = {
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m", # Удерживаем модель в памяти
            "options": {
                "temperature": 0.0,
                "num_predict": 15,   # Ограничиваем генерацию 15 токенами для максимальной скорости
                "num_ctx": 512       # Уменьшаем размер контекста для ускорения работы attention-слоев
            }
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers={'Content-Type': 'application/json', 'Connection': 'keep-alive'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                answer = res_data.get("response", "").strip()
                
            best_topic = "Другое" if lang == "ru" else "Other"
            for t in topics:
                if t.lower() in answer.lower():
                    best_topic = t
                    break
                    
            return {
                "post_id": post_id,
                "topic": best_topic,
                "score": 1.0
            }
            
        except Exception as e:
            return {
                "post_id": post_id,
                "topic": "Ошибка анализа" if lang == "ru" else "Analysis Error",
                "score": 0.0
            }

    # 3. Параллельная обработка пачки запросов через ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_post, item) for item in valid_posts]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Анализ тем (Ollama Optimized)"):
            results.append(future.result())
            
    return results

def analyze_posts_batch(posts_data, topic_pipeline, topics, hypothesis_template, device):
    results = []
    texts_for_gpu = []
    metadata_for_gpu = []
    
    for post_id, text in posts_data:
        text_str = str(text) if text is not None else ""
        
        if not text_str or len(text_str.strip()) < 10:
            results.append({
                "post_id": post_id,
                "topic": "Без категории" if "текст" in hypothesis_template else "Uncategorized",
                "score": 0.0
            })
            continue
            
        texts_for_gpu.append(text_str[:2000])
        metadata_for_gpu.append(post_id)
        
    combined_data = list(zip(texts_for_gpu, metadata_for_gpu))
    combined_data.sort(key=lambda x: len(x[0]))
    
    texts_for_gpu = [x[0] for x in combined_data]
    metadata_for_gpu = [x[1] for x in combined_data]
    
    BATCH_SIZE = 32 if device == 0 else 8
    num_batches = (len(texts_for_gpu) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(texts_for_gpu), BATCH_SIZE), total=num_batches, desc="Анализ тем (Transformers)"):
        batch_texts = texts_for_gpu[i:i+BATCH_SIZE]
        batch_ids = metadata_for_gpu[i:i+BATCH_SIZE]
        
        try:
            out_batch = topic_pipeline(
                batch_texts, 
                candidate_labels=topics, 
                hypothesis_template=hypothesis_template,
                multi_label=False, 
                batch_size=len(batch_texts)
            )
            
            if isinstance(out_batch, dict):
                out_batch = [out_batch]
                
            for j, out in enumerate(out_batch):
                post_id = batch_ids[j]
                best_topic = out['labels'][0]
                best_score = out['scores'][0]
                
                if best_score < 0.2:
                    best_topic = "Другое" if "текст" in hypothesis_template else "Other"

                results.append({
                    "post_id": post_id,
                    "topic": best_topic,
                    "score": round(best_score, 4)
                })
                
        except Exception as e:
            print(f"\nОшибка в батче: {e}")
            for post_id in batch_ids:
                results.append({
                    "post_id": post_id,
                    "topic": "Ошибка анализа" if "текст" in hypothesis_template else "Analysis Error",
                    "score": 0.0
                })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", default="telegram_export.db")
    parser.add_argument("--lang", choices=["ru", "en"], default="ru")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Ошибка: не найден файл БД {args.db_path}")
        return

    print("Выберите режим работы:")
    print("1 - HuggingFace Transformers (Zero-shot pipeline)")
    print("2 - Ollama (Локальная модель qwen2.5:3b - Многопоточная версия)")
    
    while True:
        mode = input("Введите 1 или 2: ").strip()
        if mode in ['1', '2']:
            break
        print("Некорректный ввод. Попробуйте еще раз.")

    if args.lang == "ru":
        hypothesis_template = "Этот текст относится к категории: {}."
        TOPICS = [
            "Политика и общество", "Спорт", "Экономика и инвестиции", 
            "Технологии и IT", "Наука", "Кино, сериалы и искусство", 
            "Видеоигры, киберспорт и стриминг", "Юмор и мемы", "Авто и транспорт", 
            "Здоровье и медицина", "Происшествия и криминал", 
            "Личная жизнь, дневник и размышления"
        ]
    else:
        hypothesis_template = "This text is about {}."
        TOPICS = [
            "Politics and Society", "Sports", "Economy and Investing", 
            "Technology and IT", "Science", "Movies, Series and Art", 
            "Video Games, Esports and Streaming", "Humor and Memes", "Auto and Transport", 
            "Health and Medicine", "Crime and Accidents", 
            "Personal Life, Diary and Reflections"
        ]

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT post_id, text FROM telegram_posts") 
        posts_rows = cursor.fetchall()
    except Exception as e:
        print(f"Ошибка чтения БД: {e}")
        conn.close()
        return

    if mode == '1':
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            print("Ошибка импорта. Установите: pip install transformers torch")
            sys.exit(1)
            
        device = 0 if torch.cuda.is_available() else -1
        MODEL_NAME = "cointegrated/rubert-base-cased-nli-twoway" if args.lang == "ru" else "facebook/bart-large-mnli"
        
        print(f"Загрузка модели {MODEL_NAME}...")
        topic_pipeline = pipeline("zero-shot-classification", model=MODEL_NAME, device=device)
        final_data = analyze_posts_batch(posts_rows, topic_pipeline, TOPICS, hypothesis_template, device)
        table_name = "PostTopics"
        
    else:
        print("Подключение к Ollama (qwen2.5:3b)...")
        # max_workers=8 по умолчанию. Если видеокарта мощная (например, RTX 3080/4070+), можно поставить max_workers=16
        final_data = analyze_posts_ollama(posts_rows, TOPICS, args.lang, max_workers=8)
        table_name = "PostTopics"

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f'''
    CREATE TABLE {table_name} (
        post_id INTEGER,
        topic TEXT,
        confidence_score REAL
    )
    ''')
    
    db_data = [(item['post_id'], item['topic'], item['score']) for item in final_data]
    cursor.executemany(f"INSERT INTO {table_name} (post_id, topic, confidence_score) VALUES (?, ?, ?)", db_data)
    
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_topic ON {table_name}(topic)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_post_id ON {table_name}(post_id)")
    
    conn.commit()
    print(f"Готово! Результаты сохранены в таблицу {table_name} базы данных {args.db_path}")
    conn.close()

if __name__ == '__main__':
    main()