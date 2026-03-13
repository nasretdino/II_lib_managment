# II Lib Management — NotebookLM-style Backend

> Учебный бэкенд для приложения-аналога NotebookLM: загрузка небольших документов, GraphRAG-поиск по ним (LightRAG) и чат с мультиагентной системой ARCADE на базе LangGraph.

---

## Оглавление

- [Описание](#описание)
- [Стек технологий](#стек-технологий)
- [Архитектура](#архитектура)
  - [ARCADE Pipeline (LangGraph)](#arcade-pipeline-langgraph)
  - [SSE и Nginx](#sse-и-nginx)
- [Структура проекта](#структура-проекта)
- [Описание каждого файла](#описание-каждого-файла)
  - [Инфраструктура (корень)](#инфраструктура-корень)
  - [src/core — Ядро приложения](#srccore--ядро-приложения)
  - [src/db — Слой базы данных](#srcdb--слой-базы-данных)
  - [src/modules/users — Пользователи](#srcmodulesusers--пользователи)
  - [src/modules/documents — Документы](#srcmodulesdocuments--документы)
  - [src/modules/rag — LightRAG](#srcmodulesrag--lightrag)
  - [src/modules/agents — Мультиагентная система (ARCADE)](#srcmodulesagents--мультиагентная-система-arcade)
  - [src/modules/chat — Чат и SSE-стриминг](#srcmoduleschat--чат-и-sse-стриминг)
- [API-эндпоинты](#api-эндпоинты)
- [Переменные окружения](#переменные-окружения)
- [Запуск](#запуск)

---

## Описание

Бэкенд реализует следующую функциональность (без фронтенда):

| Зона фронтенда | Что покрывает бэкенд |
|---|---|
| **Левая панель** — загрузка файлов | Приём PDF/TXT/DOCX, парсинг через MarkItDown / PyMuPDF, разбиение на чанки, индексация через LightRAG |
| **Правая панель** — чат | SSE-стриминг ответа, мультиагентный ARCADE-пайплайн (6 узлов LangGraph) с валидацией через PydanticAI |
| **Профиль** | CRUD пользователей, история диалогов |

### Ключевые принципы

- **ARCADE** (Ask → Retrieve → Analyze → Critique → Decide → Emit) — вместо одного гигантского промпта задача разбита на 6 специализированных узлов LangGraph, каждый со своей ответственностью.
- **LightRAG** — библиотека для GraphRAG, которая берёт на себя построение графа знаний, извлечение сущностей/связей и гибридный поиск (vector + graph).
- **Хранилища по ролям** — PostgreSQL + pgvector хранит реляционные данные и embeddings, Neo4j хранит граф Entities/Relationships.
- **Простота** — учебный проект без брокеров сообщений, очередей, Kubernetes. Docker Compose поднимает PostgreSQL + Neo4j + бэкенд, и всё работает.
- **Потоковая передача** — Server-Sent Events для вывода ответов токен за токеном, включая промежуточные статусы (routing, retrieval, analysis, critique).

---

## Стек технологий

| Категория | Технология |
|---|---|
| Язык | Python 3.14 |
| Пакетный менеджер | Poetry |
| Web-framework | FastAPI |
| Reverse proxy | Nginx (с оптимизацией для SSE) |
| ORM | SQLAlchemy 2 (async) |
| DB-драйвер | asyncpg |
| Миграции | Alembic |
| База данных | PostgreSQL + pgvector + Neo4j + MinIO |
| Логирование | Loguru |
| Парсинг документов | MarkItDown / PyMuPDF |
| Разбиение на чанки | Semantic Router / text_splitter из LangChain |
| GraphRAG | LightRAG |
| Оркестрация агентов | LangGraph (6 узлов ARCADE) |
| Валидация ответов LLM | PydanticAI |
| Стриминг | Server-Sent Events (SSE) |
| Контейнеризация | Docker, Docker Compose |
| LLM-провайдеры | API (OpenAI / Anthropic) — без локальных моделей |

---

## Архитектура

```
┌─────────┐     ┌─────────────┐     ┌─────────────────────────────────────┐
│  Client │────▶│    Nginx     │────▶│  Gunicorn + Uvicorn Workers          │
│ (SSE)   │◀────│ (SSE-ready) │◀────│                                     │
└─────────┘     └─────────────┘     │  ┌─────────────────────────────────┐│
                                    │  │  FastAPI Application            ││
                                    │  │                                 ││
                                    │  │  /users    → UserService        ││
                                    │  │  /docs     → DocService         ││
                                    │  │  /chat/sse → LangGraph ARCADE ─┐││
                                    │  │                                │││
                                    │  │  ┌──── ARCADE (6 узлов) ──────┤││
                                    │  │  │                            │││
                                    │  │  │  Ask ──▶ Retrieve ──▶ Analyze│
                                    │  │  │                       │    │││
                                    │  │  │  Emit ◀── Decide ◀── Critique│
                                    │  │  │            │               │││
                                    │  │  │            └── retry ──▶ ──┘││
                                    │  │  └────────────────────────────┘││
                                    │  └────────────────────────────────┘│
                                    └────────────────┬──────────────────┘
                                                     │
                                    ┌────────────────▼──────────────────┐
                                    │   PostgreSQL + pgvector           │
                                    │                                   │
                                    │  users, documents, chat_sessions  │
                                    │  chat_messages, vector embeddings │
                                    └────────────────┬──────────────────┘
                                                     │
                                    ┌────────────────▼──────────────────┐
                                    │             Neo4j                 │
                                    │                                   │
                                    │  entity graph + relationships     │
                                    │  (Graph RAG knowledge layer)      │
                                    └───────────────────────────────────┘
```

### ARCADE Pipeline (LangGraph)

Вместо одного гигантского промпта — **6 специализированных узлов** в графе LangGraph:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph                         │
│                                                                     │
│  ┌───────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  ASK  │───▶│ RETRIEVE │───▶│ ANALYZE  │───▶│ CRITIQUE │         │
│  │Router │    │Researcher│    │ Analyst  │    │  Critic  │         │
│  └───────┘    └──────────┘    └──────────┘    └────┬─────┘         │
│       │                            ▲               │               │
│       │                            │          ┌────▼─────┐         │
│       │                            └──retry───│  DECIDE  │         │
│       │                                       └────┬─────┘         │
│       │                                            │ ok            │
│       │                                       ┌────▼─────┐         │
│       │                                       │   EMIT   │         │
│       │                                       └──────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

| Узел | Роль | Описание |
|---|---|---|
| **Ask** (Router) | Маршрутизация | Классифицирует входящий запрос: поиск по конкретному документу, междокументная аналитика или генерация подкаста. Активирует соответствующий подграф. |
| **Retrieve** (Researcher) | Извлечение данных | Агент-исследователь использует tool-calling для поиска через LightRAG (vector + graph). Инструменты возвращают строго валидированные данные через Pydantic-схемы, а не «сырые» тексты. |
| **Analyze** (Analyst) | Синтез ответа | Принимает собранные данные и формирует первичный проект ответа, базируясь исключительно на фактах из базы. |
| **Critique** (Critic) | Оценка и рефлексия | Паттерн Reflection (Generator-Evaluator). Критик **не имеет доступа к функции генерации контента** — его единственная цель: проверить черновик на галлюцинации, логические разрывы и соответствие исходным сноскам (citations). |
| **Decide** | Условный переход | Если Критик находит изъян → формирует корректирующую инструкцию и направляет поток обратно на Retrieve или Analyze. Если ответ ок → переход к Emit. |
| **Emit** | Вывод | Форматирование финального ответа и отправка клиенту в потоковом режиме (SSE). |


## Структура проекта

```
.
├── docker-compose.yml           # Оркестрация: db, nginx, backend
├── .env.example                 # Шаблон переменных окружения
├── .env                         # Локальные переменные (git-ignored)
├── nginx/
│   └── nginx.conf               # Reverse proxy + оптимизация SSE
├── Dockerfile                   # Сборка образа бэкенда
├── gunicorn_conf.py             # Gunicorn (воркеры, таймауты)
├── pyproject.toml               # Poetry: зависимости
├── poetry.lock                  # Lockfile
├── alembic.ini                  # Конфигурация Alembic
├── migrations/
│   ├── env.py                   # Настройка async-миграций
│   ├── script.py.mako           # Шаблон миграций
│   └── versions/                # Файлы миграций
│
└── src/
    ├── main.py                  # Точка входа: FastAPI app, lifespan, роутеры
    │
    ├── core/
    │   ├── __init__.py
    │   ├── config.py            # Pydantic Settings: DB, API-ключи, LLM
    │   ├── exceptions.py        # Кастомные HTTP-исключения
    │   └── logging.py           # Конфигурация Loguru
    │
    ├── db/
    │   ├── __init__.py
    │   ├── base_model.py        # DeclarativeBase
    │   ├── base_dao.py          # Универсальный CRUD DAO
    │   └── session.py           # Engine, session, get_db()
    │
    └── modules/
        ├── __init__.py
        │
        ├── users/               
        │   ├── models.py        # Таблица users
        │   ├── schemas.py       # UserCreate, UserUpdate, UserRead, UserFilter
        │   ├── dao.py           # UserDAO(BaseDAO)
        │   ├── services.py      # Бизнес-логика
        │   ├── routers.py       # CRUD /users
        │   └── dependencies.py  # DI
        │
        ├── documents/           # Загрузка и обработка документов
        │   ├── models.py        # Таблица documents
        │   ├── schemas.py       # DocumentUpload, DocumentRead
        │   ├── dao.py           # DocumentDAO
        │   ├── services.py      # Оркестрация: upload → parse → chunk → index
        │   ├── routers.py       # /documents (upload, list, delete)
        │   ├── dependencies.py  # DI
        │   ├── parser.py        # Парсинг файлов через MarkItDown / PyMuPDF
        │   └── chunking.py      # Разбиение на чанки (Semantic Router / text_splitter)
        │
        ├── rag/                 # LightRAG: индексация и поиск
        │   ├── __init__.py
        │   ├── schemas.py       # SearchResult, ChunkRead
        │   ├── services.py      # RagService — обёртка над LightRAG
        │   └── dependencies.py  # DI
        │
        ├── agents/              # Мультиагентная система (ARCADE)
        │   ├── __init__.py
        │   ├── state.py         # TypedDict State для LangGraph
        │   ├── tools.py         # Tool-функции (search через LightRAG)
        │   ├── nodes/           # 6 узлов графа
        │   │   ├── __init__.py
        │   │   ├── router.py    # Ask — маршрутизация запроса
        │   │   ├── researcher.py# Retrieve — извлечение данных
        │   │   ├── analyst.py   # Analyze — синтез ответа (PydanticAI)
        │   │   ├── critic.py    # Critique — проверка на галлюцинации (PydanticAI)
        │   │   ├── decision.py  # Decide — условный переход
        │   │   └── emitter.py   # Emit — форматирование и отправка
        │   └── workflow.py      # StateGraph: связи узлов, условные рёбра
        │
        └── chat/                # Чат и стриминг
            ├── models.py        # chat_sessions, chat_messages
            ├── schemas.py       # ChatRequest, ChatEvent, SessionRead
            ├── dao.py           # ChatDAO
            ├── services.py      # ChatService — запуск workflow, SSE
            ├── routers.py       # POST /chat/stream, GET /chat/sessions
            └── dependencies.py  # DI
```



### DocumentService — оркестрация пайплайна загрузки:

```
1. Сохранить файл на диск
2. Создать запись в БД (status="pending")
3. Вызвать parser.extract_text() → text_content
4. Вызвать chunking.split_into_chunks() → список чанков
5. Передать чанки в RagService для индексации через LightRAG
6. Обновить status="ready"
```

Так как проект учебный и файлы маленькие, весь пайплайн выполняется синхронно в рамках запроса — без очередей задач.


#### `documents/chunking.py`

Разбиение текста на чанки с использованием **Semantic Router** или standalone **text_splitter** из LangChain (можно вытащить только модуль `langchain_text_splitters`, не таща весь фреймворк):

```python
async def split_into_chunks(
    text: str,
    doc_summary: str,    # краткое описание документа (от LLM)
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    1. Разбивает текст с учётом структуры:
       - Semantic Router — семантическое разбиение по смыслу
       - Или MarkdownHeaderTextSplitter / RecursiveCharacterTextSplitter
         из langchain_text_splitters — бьёт текст, ориентируясь
         на Markdown-заголовки и структуру предложений
    2. Contextual Enrichment: к каждому чанку в начало добавляется
       краткое описание всего документа, чтобы эмбеддинг чанка
       учитывал глобальный контекст
    """
```

---

### `src/modules/rag` — LightRAG

Индексация и граф-поиск по документам через библиотеку **LightRAG**.

LightRAG берёт на себя:
- Построение графа знаний (сущности + связи) из текста
- Создание эмбеддингов и их хранение
- Гибридный поиск: vector similarity + graph traversal
- Multi-hop reasoning по графу

Нет необходимости вручную писать SQL для graph traversal, CTE-запросы и embedder — всё инкапсулировано в LightRAG.

#### `rag/schemas.py`

```python
class SearchResult(BaseModel):
    """Результат поиска, возвращаемый агентам"""
    context_text: str          # Собранный контекст из LightRAG
    sources: list[str]         # Источники (имена документов / chunk IDs)
    mode: str                  # "naive" | "local" | "global" | "hybrid"
```

#### `rag/services.py`

`RagService` — тонкая обёртка над LightRAG:

```python
class RagService:
    def __init__(self):
        self.rag = LightRAG(
            working_dir="./rag_storage",
            llm_model_func=...,       # OpenAI / Anthropic
            embedding_func=...,        # text-embedding-3-small
        )

    async def index_document(self, doc_id: int, chunks: list[str]) -> None:
        """Вставить чанки в LightRAG для индексации (граф + векторы)."""
        for chunk in chunks:
            await self.rag.ainsert(chunk)

    async def search(self, query: str, mode: str = "hybrid") -> SearchResult:
        """
        Поиск через LightRAG:
        - "naive"  — только vector similarity
        - "local"  — поиск по локальному контексту сущностей
        - "global" — глобальный обзор по всему графу
        - "hybrid" — комбинация local + global
        """
        result = await self.rag.aquery(query, param=QueryParam(mode=mode))
        return SearchResult(context_text=result, ...)
```

---

### `src/modules/agents` — Мультиагентная система (ARCADE)

Реализация **ARCADE** через **LangGraph** с **6 специализированными узлами**.

#### `agents/state.py`

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str               # Вопрос пользователя
    query_type: str             # Тип запроса (от Router): "search" | "analytics" | "podcast"
    context: str                # RAG-контекст из LightRAG
    draft_answer: str           # Текущий черновик ответа (от Analyst)
    critique: str               # Замечания Critic
    correction: str             # Корректирующая инструкция (от Decide)
    is_approved: bool           # Принят ли ответ
    iteration: int              # Номер итерации цикла
    max_iterations: int         # Макс. число итераций
    final_answer: str           # Финальный ответ
    events: list[dict]          # SSE-события для стриминга
```

#### `agents/tools.py`

Tool-функции, которые агенты могут вызывать через LangGraph tool-calling:

```python
async def search_knowledge(query: str, mode: str = "hybrid") -> str:
    """
    Инструмент для Researcher: поиск по базе знаний через LightRAG.
    Возвращает валидированный через Pydantic контекст, а не «сырой» текст.
    """
```

#### `agents/nodes/router.py` — Ask

```python
async def router_node(state: AgentState) -> AgentState:
    """
    Маршрутизатор: классифицирует запрос пользователя.
    Определяет query_type:
    - "search"    — поиск по конкретному документу
    - "analytics" — междокументная аналитика
    - "podcast"   — генерация подкаста
    Активирует соответствующий подграф.
    """
```

#### `agents/nodes/researcher.py` — Retrieve

```python
async def researcher_node(state: AgentState) -> AgentState:
    """
    Агент-исследователь:
    1. На основе query_type выбирает режим поиска LightRAG (local/global/hybrid)
    2. Использует tool-calling для генерации запросов к LightRAG
    3. Результаты валидируются через Pydantic-схемы
    4. Записывает context в state
    """
```

#### `agents/nodes/analyst.py` — Analyze

```python
from pydantic_ai import Agent

analyst_agent = Agent(
    model="openai:gpt-4o",
    result_type=AnalystResponse,
    system_prompt="Ты — аналитик. На основе контекста дай точный ответ...",
)

async def analyst_node(state: AgentState) -> AgentState:
    """
    1. Получает вопрос + контекст + (если есть) замечания критика и корректирующую инструкцию
    2. Формирует ответ через PydanticAI (гарантия валидного JSON)
    3. Записывает draft_answer в state
    4. SSE-событие: {"type": "analyst_thinking", "data": ...}
    """
```

#### `agents/nodes/critic.py` — Critique

```python
critic_agent = Agent(
    model="anthropic:claude-sonnet-4-20250514",
    result_type=CriticResponse,    # {"is_approved": bool, "critique": str, "citations_valid": bool}
    system_prompt="Ты — критик. Проверь ответ на галлюцинации...",
)

async def critic_node(state: AgentState) -> AgentState:
    """
    Паттерн Reflection (Generator-Evaluator):
    1. Получает draft_answer + оригинальный контекст
    2. Проверяет: галлюцинации, логические разрывы, соответствие сноскам
    3. НЕ имеет доступа к генерации контента — только оценка
    4. SSE-событие: {"type": "critic_review", "data": ...}

    Намеренно другая модель (Anthropic vs OpenAI) —
    уменьшает вероятность одинаковых галлюцинаций.
    """
```

#### `agents/nodes/decision.py` — Decide

```python
async def decision_node(state: AgentState) -> AgentState:
    """
    Узел условного перехода:
    - Если Критик нашёл изъян → формирует корректирующую инструкцию,
      направляет поток обратно на Retrieve или Analyze
    - Если ответ ок или достигнут max_iterations → переход к Emit
    """
```

#### `agents/nodes/emitter.py` — Emit

```python
async def emitter_node(state: AgentState) -> AgentState:
    """
    Форматирование финального ответа и подготовка для SSE-стриминга.
    Записывает final_answer в state.
    """
```

## API-эндпоинты

### Users

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `GET` | `/users/` | Список пользователей |
| `GET` | `/users/{id}` | Профиль пользователя |
| `POST` | `/users/` | Создать пользователя |
| `PATCH` | `/users/{id}` | Обновить профиль |
| `DELETE` | `/users/{id}` | Удалить пользователя |

### Documents

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `POST` | `/documents/upload` | Загрузить файл (PDF/TXT/DOCX) |
| `GET` | `/documents/` | Список документов пользователя |
| `GET` | `/documents/{id}` | Метаданные документа |
| `DELETE` | `/documents/{id}` | Удалить документ |

### Chat

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `POST` | `/chat/stream` | SSE-стриминг ответа (ARCADE) |
| `GET` | `/chat/sessions` | Список сессий пользователя |
| `GET` | `/chat/sessions/{id}/messages` | История сообщений |

---

## Переменные окружения

| Переменная | Описание | Пример |
|---|---|---|
| `ENV` | Окружение | `dev` / `prod` |
| `DB__HOST` | Хост PostgreSQL (`db` для Docker, `localhost` для локальной разработки) | `db` |
| `DB__PORT` | Порт PostgreSQL | `5432` |
| `DB__USER` | Пользователь БД | `postgres` |
| `DB__PASSWORD` | Пароль БД | `postgres` |
| `DB__NAME` | Имя базы данных | `lib_management` |
| `DB__ECHO` | SQL-логирование (опционально; если не задан — `true` в dev, `false` в prod) | — |
| `DB__POOL_SIZE` | Размер пула соединений | `20` |
| `DB__MAX_OVERFLOW` | Макс. дополнительных соединений | `10` |
| `DB__POOL_RECYCLE` | Время жизни соединения (сек) | `1800` |
| `RAG__GRAPH_STORAGE` | Graph backend для LightRAG (`Neo4JStorage` или `NetworkXStorage`) | `Neo4JStorage` |
| `NEO4J__HOST` | Хост Neo4j (`neo4j` в Docker Compose) | `neo4j` |
| `NEO4J__PORT` | Bolt-порт Neo4j | `7687` |
| `NEO4J__USER` | Пользователь Neo4j | `neo4j` |
| `NEO4J__PASSWORD` | Пароль Neo4j | `neo4jpassword` |
| `NEO4J__DATABASE` | Имя базы Neo4j | `neo4j` |
| `NEO4J__WORKSPACE` | Workspace-лейбл графа в LightRAG | `default` |
| `STORAGE__MODE` | Хранилище файлов: `local` (папка проекта) или `minio` (S3) | `minio` |
| `STORAGE__LOCAL_PATH` | Путь к папке при `STORAGE__MODE=local` | `uploads` |
| `MINIO__ENDPOINT` | Адрес MinIO S3 API (при `STORAGE__MODE=minio`) | `minio:9000` |
| `MINIO__ACCESS_KEY` | Access key для MinIO | `minioadmin` |
| `MINIO__SECRET_KEY` | Secret key для MinIO | `minioadmin` |
| `MINIO__BUCKET_NAME` | Бакет для хранения файлов | `documents` |
| `MINIO__USE_SSL` | Использовать HTTPS к MinIO | `false` |
| `LLM__PROVIDER` | LLM-провайдер: `gemini` или `ollama` | `gemini` |
| `LLM__API_KEY` | API-ключ (обязателен для gemini) | `AI...` |
| `LLM__OLLAMA_HOST` | Хост Ollama (при `LLM__PROVIDER=ollama`) | `http://localhost:11434` |
| `LLM__MODEL_NAME` | Модель для генерации | `gemini-2.5-flash` |
| `LLM__EMBEDDING_MODEL` | Модель для эмбеддингов | `gemini-embedding-001` |
| `LLM__EMBEDDING_DIM` | Размерность вектора | `768` |
| `LLM__CHUNK_TOKEN_SIZE` | Размер чанка (токены) | `1200` |
| `LLM__CHUNK_OVERLAP_TOKEN_SIZE` | Перекрытие чанков (токены) | `100` |

---

## Запуск

### Dev (локальная разработка)

Инфраструктурные сервисы запускаются в Docker, приложение — локально.

```bash
# 1. Скопировать шаблон переменных
cp .env.example .env

# 2. Выставить для локальной разработки:
#    ENV=dev
#    DB__HOST=localhost      (← важно! не "db")
#    NEO4J__HOST=localhost   (← важно! не "neo4j")
#    MINIO__ENDPOINT=localhost:9000  (← если используете MinIO)
#    LLM__API_KEY=<ваш ключ Gemini API>

# 3. Выбрать режим хранения файлов (один из двух):

#    Вариант A — локальное хранилище (без MinIO):
#    STORAGE__MODE=local
#    Файлы будут сохраняться в папку uploads/ проекта.
#    MinIO запускать не нужно.

#    Вариант B — MinIO (S3-совместимое хранилище):
#    STORAGE__MODE=minio
#    MINIO__ENDPOINT=localhost:9000

# 4. (Опционально) Выбрать graph storage:
#    RAG__GRAPH_STORAGE=Neo4JStorage    ← полноценный граф (нужен Neo4j)
#    RAG__GRAPH_STORAGE=NetworkXStorage ← локальный fallback (Neo4j не нужен)

# 5. Поднять инфраструктуру в Docker
#    Минимальный набор — только PostgreSQL:
docker compose up db -d

#    Если используете Neo4j (RAG__GRAPH_STORAGE=Neo4JStorage):
docker compose up db neo4j -d

#    Если используете MinIO (STORAGE__MODE=minio):
docker compose up db neo4j minio minio-init -d

# 6. Прогнать миграции
poetry run alembic upgrade head

# 7. Запустить сервер
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 8. Тесты (БД не нужна — тесты используют SQLite in-memory,
#    MinIO и RAG мокаются, данные проекта не затрагиваются)
poetry run pytest tests/
```

**Минимальный dev-сетап** (без MinIO и Neo4j):

```env
ENV=dev
DB__HOST=localhost
DB__PORT=5432
DB__USER=postgres
DB__PASSWORD=postgres
DB__NAME=lib_management
STORAGE__MODE=local
RAG__GRAPH_STORAGE=NetworkXStorage
LLM__API_KEY=<ваш ключ>
```

```bash
docker compose up db -d
poetry run alembic upgrade head
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

В dev-режиме (`ENV=dev`):

- `debug=True` в FastAPI
- `/docs` и `/redoc` доступны
- SQL-запросы логируются (`db_echo=True`)
- Loguru на уровне `DEBUG`

### Prod (Docker Compose)

Все сервисы запускаются через Docker Compose.

```bash
# 1. Настроить .env:
#    ENV=prod
#    DB__HOST=db                    (← имя сервиса в docker-compose)
#    DB__PASSWORD=<надёжный пароль>
#    NEO4J__HOST=neo4j              (← имя сервиса)
#    NEO4J__PASSWORD=<надёжный пароль>
#    MINIO__ENDPOINT=minio:9000     (← имя сервиса)
#    STORAGE__MODE=minio
#    RAG__GRAPH_STORAGE=Neo4JStorage
#    LLM__API_KEY=<ваш ключ>

# 2. Запустить всё
docker compose up -d --build
```

Docker Compose поднимет:
- **db** — PostgreSQL 17 с pgvector
- **neo4j** — Neo4j 5.26 (граф знаний)
- **minio** — S3-совместимое объектное хранилище
- **minio-init** — создание бакета `documents`
- **migrate** — однократный запуск `alembic upgrade head`
- **app** — FastAPI через Uvicorn