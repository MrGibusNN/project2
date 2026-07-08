import sqlite3
import os
import sys
import json
import urllib.request
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def analyze_comments_ollama(comments_data, max_workers=8):
    results = []
    url = "http://localhost:11434/api/generate"
    
    # 1. Отсеиваем слишком короткие комментарии сразу, чтобы не нагружать модель
    valid_comments = []
    for com_id, post_id, com_text, post_text in comments_data:
        c_text_str = str(com_text).strip() if com_text else ""
        p_text_str = str(post_text).strip() if post_text else ""
        
        if len(c_text_str) < 2:
            results.append((com_id, post_id, "neutral", 0.0))
        else:
            valid_comments.append((com_id, post_id, c_text_str, p_text_str))

    # 2. Функция для отправки одного запроса в Ollama
    def process_single_comment(item):
        com_id, post_id, c_text_str, p_text_str = item
        
        prompt = f"""Проанализируй отношение комментатора к содержанию поста.

Текст поста:
{p_text_str[:1500]}

Комментарий:
{c_text_str[:500]}

Определи тональность комментария ИМЕННО по отношению к автору или тексту поста (поддерживает/согласен, критикует/возмущается, или нейтрален/задает вопрос/оффтоп).
Выбери строго один вариант: positive, negative или neutral.
Ответь ТОЛЬКО одним словом на английском языке."""

        payload = {
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",  # Удерживаем модель в памяти
            "options": {
                "temperature": 0.0,
                "num_predict": 5,    # Нам нужно всего 1 слово, ограничиваем генерацию ради скорости
                "num_ctx": 1024      # Оптимальный размер контекста для поста и коммента
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
                answer = res_data.get("response", "").strip().lower()
                
            # Защита от лишних слов в ответе модели
            if "positive" in answer or "позитив" in answer or "соглас" in answer:
                status = "positive"
            elif "negative" in answer or "негатив" in answer or "критик" in answer:
                status = "negative"
            else:
                status = "neutral"
                
            return (com_id, post_id, status, 1.0)
            
        except Exception as e:
            return (com_id, post_id, 'error', 0.0)

    # 3. Параллельный запуск запросов через ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_comment, item) for item in valid_comments]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Анализ отношения (Ollama Multi-thread)"):
            results.append(future.result())
            
    return results

def analyze_comments_batch(comments_data, classifier, device):
    results = []
    texts_for_gpu = []
    metadata_for_gpu = []

    for com_id, post_id, com_text, _ in comments_data:
        com_text_str = str(com_text).strip() if com_text else ""
        
        if len(com_text_str) < 2:
            results.append((com_id, post_id, "neutral", 0.0))
            continue
            
        texts_for_gpu.append(com_text_str[:512]) 
        metadata_for_gpu.append((com_id, post_id))

    combined_data = list(zip(texts_for_gpu, metadata_for_gpu))
    combined_data.sort(key=lambda x: len(x[0]))
    
    texts_for_gpu = [x[0] for x in combined_data]
    metadata_for_gpu = [x[1] for x in combined_data]

    BATCH_SIZE = 32 if device == 0 else 8 
    
    for i in tqdm(range(0, len(texts_for_gpu), BATCH_SIZE), desc="Анализ (Transformers)"):
        batch_texts = texts_for_gpu[i:i+BATCH_SIZE]
        batch_meta = metadata_for_gpu[i:i+BATCH_SIZE]
        
        try:
            out_batch = classifier(batch_texts, batch_size=len(batch_texts))
            
            if isinstance(out_batch, dict):
                out_batch = [out_batch]
                
            for j, out in enumerate(out_batch):
                com_id, post_id = batch_meta[j]
                
                best_label = out['label']
                best_score = out['score']
                
                if best_label == 'POSITIVE':
                    status = "positive"
                elif best_label == 'NEGATIVE':
                    status = "negative"
                else:
                    status = "neutral"
                    
                results.append((com_id, post_id, status, round(best_score, 4)))
                
        except Exception:
            for meta in batch_meta:
                results.append((meta[0], meta[1], 'error', 0.0))

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", default="telegram_export.db")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Ошибка: не найден файл БД {args.db_path}")
        sys.exit(1)

    print("Выберите режим работы (Анализ тональности комментариев):")
    print("1 - HuggingFace (rubert-base-cased-sentiment, анализ только текста коммента)")
    print("2 - Ollama (qwen2.5:3b, контекстный анализ отношения комментатора к посту)")
    
    while True:
        mode = input("Введите 1 или 2: ").strip()
        if mode in ['1', '2']:
            break
        print("Некорректный ввод. Попробуйте еще раз.")

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    # Извлекаем не только коммент, но и связанный текст поста для контекста
    try:
        cursor.execute('''
            SELECT c.id, c.post_id, c.text, p.text 
            FROM telegram_comments c
            LEFT JOIN telegram_posts p ON c.post_id = p.post_id
        ''')
        comments = cursor.fetchall()
    except Exception as e:
        print(f"Ошибка чтения БД: {e}. Проверьте структуру таблиц.")
        conn.close()
        return

    if mode == '1':
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            print("Ошибка импорта. Установите: pip install transformers torch")
            sys.exit(1)
            
        device = 0 if torch.cuda.is_available() else -1
        MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
        print(f"Загрузка модели {MODEL_NAME}...")
        
        classifier = pipeline("text-classification", model=MODEL_NAME, device=device)
        final_results = analyze_comments_batch(comments, classifier, device)
        table_name = "CommentRelation"
        
    else:
        print("Подключение к Ollama (qwen2.5:3b)...")
        # По умолчанию 8 потоков. Если компьютер мощный, можно изменить max_workers=16 в вызове ниже
        final_results = analyze_comments_ollama(comments, max_workers=8)
        table_name = "CommentRelation"

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f'''
    CREATE TABLE {table_name} (
        comment_id INTEGER,
        post_id INTEGER,
        relation TEXT,
        confidence_score REAL
    )
    ''')
    
    cursor.executemany(f'''
    INSERT INTO {table_name} (comment_id, post_id, relation, confidence_score)
    VALUES (?, ?, ?, ?)
    ''', final_results)
    
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_post_id ON {table_name}(post_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_relation ON {table_name}(relation)")
    
    conn.commit()
    print(f"Готово! Результаты сохранены в таблицу {table_name} базы данных {args.db_path}")
    conn.close()

if __name__ == '__main__':
    main()