# Раздел 2. Выбор и реализация AI/ML/DL-методов и архитектура системы

**Дисциплина:** AI Essentials: Theory and Applications  
**Соответствие критериям оценивания:**
- **Критерий 3:** Выбор и реализация AI/ML/DL-метода (Макс. 15 баллов)

---

## 2.1. Общая архитектурная схема

Система представляет собой гибридный конвейер, объединяющий глубокие нейросетевые модели для извлечения информации (NER), плотный векторный поиск (Dense Vector Retrieval на базе pgvector), оптическое распознавание символов (OCR) и генеративные языковые модели (LLM Reasoning).

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ВХОДНЫЕ ДАННЫЕ                                  │
│  1. Документы: PDF, DOCX, HTML, TXT    │  2. Скриншот экрана (Hotkey)  │
└───────────────────┬────────────────────┴──────────────────┬────────────┘
                    │                                       │
                    ▼                                       ▼
┌────────────────────────────────────────┐  ┌────────────────────────────┐
│      Парсеры (PyMuPDF, docx, HTML)     │  │  Tesseract OCR Service     │
│   Чанкование (400 токенов, overlap 50) │  │  (Экстракция текста экрана)│
└───────────────────┬────────────────────┘  └───────────────┬────────────┘
                    │                                       │
                    ▼                                       ▼
┌────────────────────────────────────────┐  ┌────────────────────────────┐
│       Medical NER (BioBERT) &          │  │  DeepSeek Flash API        │
│    Multilingual Fallback (XLM-R)       │  │  Дистилляция запроса       │
└───────────────────┬────────────────────┘  └───────────────┬────────────┘
                    │                                       │
                    ▼                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Векторизация: intfloat/multilingual-e5-base                │
│             Формирование префиксов "passage:" и "query:"               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PostgreSQL + pgvector (Косинусное сходство)              │
│                 SELECT * FROM chunks ORDER BY embedding <=> query      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   RAG Генерация и Синтез Ответа                        │
│   • DeepSeek Flash (Облако, низкая задержка)                           │
│   • Ollama Dolphin 8B (Локальный приватный режим, on-premise)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       ИНТЕРФЕЙСЫ ПОЛЬЗОВАТЕЛЯ                          │
│   • Streamlit Web UI (app.py)   • PySide6 Desktop Assistant Overlay    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2. Извлечение сущностей (Named Entity Recognition)

### Базовый уровень (Baseline)
- В качестве baseline-модели использована предобученная модель общего назначения **`dslim/bert-base-NER`** (обученная на корпусе CoNLL-2003: `PER`, `LOC`, `ORG`, `MISC`).
- **Недостатки baseline:** Полная неспособность распознавать специфические клинические термины (например, лекарства классифицируются как `MISC` или `ORG`, симптомы игнорируются).

### Финальная модель: Доменно-адаптированный BioBERT
- **Базовая архитектура:** `dmis-lab/biobert-v1.1` (BERT Base, инициализированный весами PubMed и PubMed Central).
- **Классификационная голова:** Линейный слой над скрытыми состояниями последнего слоя трансформера (`Linear(768, 13)`), прогнозирующий один из 13 классов схемы BIOES:
  - `O`, `B-PROBLEM`, `I-PROBLEM`, `E-PROBLEM`, `S-PROBLEM`
  - `B-TREATMENT`, `I-TREATMENT`, `E-TREATMENT`, `S-TREATMENT`
  - `B-TEST`, `I-TEST`, `E-TEST`, `S-TEST`
- **Процесс дообучения (Fine-tuning):**
  - Оптимизатор: AdamW (`lr=2e-5`, `weight_decay=0.01`).
  - Планировщик скорости обучения: Linear warmup with cosine decay.
  - Функция потерь: CrossEntropyLoss с маскированием субтокенов (`ignore_index=-100`).
  - Batch size: 16 (с накоплением градиентов gradient accumulation steps = 2).
  - Эпохи: 3 эпохи с ранней остановкой (Early Stopping по macro-F1 на валидационной выборке).

### Межъязыковая поддержка (Multilingual Fallback)
Для документов на русском и казахском языках реализован динамический роутер языка (`models/ner.py`):
1. **Казахский язык:** `yeshpanovrustem/xlm-roberta-large-ner-kazakh`.
2. **Русский язык:** `Davlan/xlm-roberta-base-ner-hrl` (многоязычный XLM-RoBERTa).
3. **Английский медицинский язык:** Наш дообученный `BioBERT-HUMADEX` (при падении или немедицинском контексте — `dslim/bert-base-NER`).

---

## 2.3. Векторные представления (Dense Retrieval)

Для преодоления ограничений ключевого поиска выбран современный энкодер:
- **Модель:** `intfloat/multilingual-e5-base` (278M параметров, 768-мерное векторное пространство).
- **Специфика E5:** Модель требует асимметричной префиксации:
  - Для фрагментов документов: `passage: <текст>`
  - Для поисковых запросов: `query: <текст>`
- **Пул эмбеддингов (Mean Pooling):**
  ```python
  def _pool_embeddings(outputs, attention_mask):
      mask = attention_mask.unsqueeze(-1)
      summed = (outputs * mask).sum(dim=1)
      counts = mask.sum(dim=1).clamp(min=1e-9)
      return torch.nn.functional.normalize(summed / counts, p=2, dim=1)
  ```
- L2-нормализация позволяет вычислять косинусное сходство через скалярное произведение, что многократно ускоряет поиск по индексам.

---

## 2.4. Хранилище векторов: PostgreSQL + pgvector

База данных построена на реляционной СУБД PostgreSQL с расширением `pgvector`:
- **Схема таблицы:**
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;

  CREATE TABLE IF NOT EXISTS documents (
      id SERIAL PRIMARY KEY,
      file_name TEXT NOT NULL,
      file_path TEXT NOT NULL,
      file_hash TEXT UNIQUE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS document_chunks (
      id SERIAL PRIMARY KEY,
      document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      embedding vector(768) NOT NULL,
      metadata JSONB
  );
  ```
- **Индексирование:** Создаётся HNSW / IVFFlat индекс для косинусного расстояния:
  ```sql
  CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  ```
- **Запрос поиска Top-K:**
  ```sql
  SELECT c.content, d.file_name, d.file_path, (1 - (c.embedding <=> %s::vector)) AS similarity
  FROM document_chunks c
  JOIN documents d ON c.document_id = d.id
  ORDER BY c.embedding <=> %s::vector
  LIMIT %s;
  ```

---

## 2.5. Комплексирование LLM: DeepSeek Flash, Ollama & Prompt Engineering

В проекте реализована гибридная схема применения больших языковых моделей (LLM):

### 1. Cloud-режим: DeepSeek Flash
- Интегрирован через официальный OpenAI-совместимый протокол (`desktop_assistant/deepseek_client.py`).
- **Сценарий 1 (Query Distillation):** Преобразование зашумлённого OCR-текста экрана в короткий поисковый запрос (5–15 слов) без вводных слов и кавычек.
- **Сценарий 2 (Overlay Chat):** Мгновенная генерация ответов на вопросы пользователя с привязкой к найденным медицинским источникам.

### 2. On-Premise режим: Ollama Dolphin 8B (`main.py`)
- Локальная модель `dolphin-llama3:8b`, запускаемая через `langchain-ollama`.
- Предназначена для сценариев повышенной конфиденциальности, когда медицинские документы нельзя передавать во внешние API.

### 3. Промпт-инжиниринг (Prompt Design)
Использованы специализированные шаблоны:
- **`ACTION_PROMPT`:** Классифицирует намерение пользователя (`rdi` — поиск документа, `rcfd` — обращение к контенту, `stop` — ответ без поиска).
- **`SUMMARY_PROMPT`:** Жестко инструктирует модель отвечать **только** на основе предоставленных фрагментов и сообщать об отсутствии информации, если факт в документах не упомянут:
  ```text
  You are given content from several documents.
  Answer the user's question using only the document content.
  If the answer isn't in the documents, say you couldn't find it.
  ```
