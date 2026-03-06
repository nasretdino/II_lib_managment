# II Lib Management — NotebookLM-style Backend

> Высоконагруженный бэкенд для приложения-аналога NotebookLM: загрузка документов, GraphRAG-поиск по ним и чат с несколькими нейросетями, которые перепроверяют друг друга по модели ARCADE.

---

## Оглавление

- [Описание](#описание)
- [Стек технологий](#стек-технологий)
- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Описание каждого файла](#описание-каждого-файла)
  - [Инфраструктура (корень)](#инфраструктура-корень)
  - [src/core — Ядро приложения](#srccore--ядро-приложения)
  - [src/db — Слой базы данных](#srcdb--слой-базы-данных)
  - [src/modules/users — Пользователи](#srcmodulesusers--пользователи)
  - [src/modules/documents — Документы](#srcmodulesdocuments--документы)
  - [src/modules/rag — GraphRAG](#srcmodulesrag--graphrag)
  - [src/modules/agents — Мультиагентная система](#srcmodulesagents--мультиагентная-система)
  - [src/modules/chat — Чат и SSE-стриминг](#srcmoduleschat--чат-и-sse-стриминг)
- [API-эндпоинты](#api-эндпоинты)
- [Переменные окружения](#переменные-окружения)
- [Запуск](#запуск)

---

## Описание

Бэкенд реализует следующую функциональность (без фронтенда):

| Зона фронтенда | Что покрывает бэкенд |
|---|---|
| **Левая панель** — загрузка файлов | Приём PDF/TXT/DOCX, парсинг, чанкинг, построение графа знаний (GraphRAG) и сохранение эмбеддингов в PostgreSQL + pgvector |
| **Правая панель** — чат | SSE-стриминг ответа, мультиагентный ворк-флоу (Analyst ↔ Critic) через LangGraph с валидацией ответов через PydanticAI |
| **Профиль** | CRUD пользователей, история диалогов |

### Ключевые принципы

- **ARCADE** (Ask → Retrieve → Analyze → Critique → Decide → Emit) — модель взаимодействия нейросетей: Analyst генерирует ответ на основе GraphRAG-контекста, Critic проверяет его на галлюцинации, цикл повторяется до консенсуса.
- **GraphRAG поверх Postgres** — вся граф-логика (сущности, связи, multi-hop reasoning) живёт внутри PostgreSQL с расширением pgvector, без Neo4j.
- **Потоковая передача** — Server-Sent Events для вывода ответов токен за токеном, включая промежуточные статусы дебатов агентов.

---

## Стек технологий

| Категория | Технология |
|---|---|
| Язык | Python 3.14 |
| Пакетный менеджер | Poetry |
| Web-framework | FastAPI |
| Reverse proxy | Nginx |
| ORM | SQLAlchemy 2 (async) |
| DB-драйвер | asyncpg |
| Миграции | Alembic |
| База данных | PostgreSQL + pgvector |
| Логирование | Loguru |
| Оркестрация агентов | LangGraph |
| Валидация ответов LLM | PydanticAI |
| Векторный поиск | pgvector (cosine similarity) |
| Стриминг | Server-Sent Events (SSE) |
| Контейнеризация | Docker, Docker Compose |
| LLM-провайдеры | API (OpenAI / Anthropic) — без локальных моделей |

---

## Архитектура

```
┌─────────┐     ┌─────────┐     ┌──────────────────────────────────────────┐
│  Client │────▶│  Nginx  │────▶│  Gunicorn + Uvicorn Workers               │
│ (SSE)   │◀────│ (proxy) │◀────│                                          │
└─────────┘     └─────────┘     │  ┌─────────────────────────────────────┐ │
                                │  │  FastAPI Application                │ │
                                │  │                                     │ │
                                │  │  /users    → UserService → UserDAO  │ │
                                │  │  /docs     → DocService  → DocDAO   │ │
                                │  │  /chat/sse → LangGraph Workflow ──┐ │ │
                                │  │                                   │ │ │
                                │  │  ┌──────────── ARCADE Loop ───────┤ │ │
                                │  │  │                                │ │ │
                                │  │  │  ┌──────────┐  ┌───────────┐  │ │ │
                                │  │  │  │ Analyst  │─▶│  Critic   │  │ │ │
                                │  │  │  │(PydanticAI) │(PydanticAI)  │ │ │
                                │  │  │  └──────────┘◀─┘───────────┘  │ │ │
                                │  │  │       ▲                       │ │ │
                                │  │  │       │  GraphRAG Context     │ │ │
                                │  │  └───────┼───────────────────────┘ │ │
                                │  └──────────┼─────────────────────────┘ │
                                └─────────────┼───────────────────────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │   PostgreSQL       │
                                    │   + pgvector       │
                                    │                    │
                                    │  users             │
                                    │  documents         │
                                    │  document_chunks   │
                                    │  entities          │
                                    │  relationships     │
                                    │  chat_sessions     │
                                    │  chat_messages     │
                                    └────────────────────┘
```

---

## Структура проекта

```
.
├── docker-compose.yml           # Оркестрация сервисов: db, nginx, backend
├── .env.example                 # Шаблон переменных окружения
├── .env                         # Локальные переменные окружения (git-ignored)
├── nginx/
│   └── nginx.conf               # Конфигурация reverse proxy + SSE
├── Dockerfile                   # Сборка образа бэкенда
├── gunicorn_conf.py             # Конфигурация Gunicorn (воркеры, таймауты)
├── pyproject.toml               # Poetry: зависимости и метаданные проекта
├── poetry.lock                  # Lockfile зависимостей
├── alembic.ini                  # Конфигурация Alembic
├── migrations/
│   ├── env.py                   # Настройка async-миграций
│   ├── script.py.mako           # Шаблон генерации файлов миграций
│   └── versions/                # Файлы миграций
│       └── ...
│
└── src/
    ├── main.py                  # Точка входа: FastAPI app, lifespan, роутеры
    │
    ├── core/                    # ── Ядро приложения ──
    │   ├── __init__.py          # Реэкспорт settings
    │   ├── config.py            # Pydantic Settings: DB, API-ключи, env
    │   ├── exceptions.py        # Кастомные HTTP-исключения
    │   └── logging.py           # Конфигурация Loguru (формат, sink, уровень)
    │
    ├── db/                      # ── Слой базы данных ──
    │   ├── __init__.py          # Реэкспорт Base, engine, get_db, BaseDAO
    │   ├── base_model.py        # DeclarativeBase + __repr__
    │   ├── base_dao.py          # Универсальный DAO: CRUD, фильтры, bulk
    │   └── session.py           # Engine, async_sessionmaker, get_db()
    │
    └── modules/                 # ── Доменные модули ──
        ├── __init__.py          # Реэкспорт всех роутеров и моделей
        │
        ├── users/               # Профили пользователей
        │   ├── __init__.py      # Реэкспорт User, user_router
        │   ├── models.py        # SQLAlchemy: таблица users
        │   ├── schemas.py       # Pydantic: UserCreate, UserUpdate, UserRead, UserFilter
        │   ├── dao.py           # UserDAO(BaseDAO) — специфичные запросы
        │   ├── services.py      # UserService — бизнес-логика
        │   ├── routers.py       # CRUD-эндпоинты /users
        │   └── dependencies.py  # DI: get_user_service
        │
        ├── documents/           # Загрузка и обработка документов
        │   ├── __init__.py      # Реэкспорт Document, document_router
        │   ├── models.py        # SQLAlchemy: таблица documents
        │   ├── schemas.py       # Pydantic: DocumentUpload, DocumentRead
        │   ├── dao.py           # DocumentDAO(BaseDAO)
        │   ├── services.py      # DocumentService — оркестрация загрузки
        │   ├── routers.py       # Эндпоинты /documents (upload, list, delete)
        │   ├── dependencies.py  # DI: get_document_service
        │   ├── parser.py        # Асинхронный парсинг PDF/TXT/DOCX → текст
        │   └── chunking.py      # Разбиение текста на чанки + Contextual Enrichment
        │
        ├── rag/                 # GraphRAG: индексация и поиск
        │   ├── __init__.py      # Реэкспорт моделей RAG
        │   ├── models.py        # SQLAlchemy: document_chunks (VECTOR), entities, relationships
        │   ├── schemas.py       # Pydantic: ChunkRead, EntityRead, SearchResult
        │   ├── dao.py           # RagDAO — векторный поиск, graph traversal
        │   ├── services.py      # RagService — оркестрация индексации
        │   ├── dependencies.py  # DI: get_rag_service
        │   ├── embedder.py      # Вызов OpenAI Embeddings API → вектор
        │   ├── graph_builder.py # Извлечение сущностей и связей из чанков (LLM)
        │   └── graph_search.py  # Multi-hop поиск: CTE + vector similarity + JOIN
        │
        ├── agents/              # Мультиагентная система (ARCADE)
        │   ├── __init__.py      # Реэкспорт run_workflow
        │   ├── state.py         # TypedDict/Pydantic State для LangGraph
        │   ├── tools.py         # Tool-функции для агентов (search_knowledge, и др.)
        │   ├── nodes/           # Узлы графа
        │   │   ├── __init__.py  # Реэкспорт analyst_node, critic_node
        │   │   ├── analyst.py   # Агент-генератор (PydanticAI) — формирует ответ
        │   │   └── critic.py    # Агент-критик (PydanticAI) — проверяет на галлюцинации
        │   └── workflow.py      # StateGraph: связи узлов, условные рёбра, цикл
        │
        └── chat/                # Чат и стриминг
            ├── __init__.py      # Реэкспорт chat_router
            ├── models.py        # SQLAlchemy: chat_sessions, chat_messages
            ├── schemas.py       # Pydantic: ChatRequest, ChatEvent, SessionRead, MessageRead
            ├── dao.py           # ChatDAO — сессии и сообщения
            ├── services.py      # ChatService — запуск workflow, сохранение истории
            ├── routers.py       # GET /chat/sessions, POST /chat/stream (SSE)
            └── dependencies.py  # DI: get_chat_service
```

---

## Описание каждого файла

### Инфраструктура (корень)

#### `docker-compose.yml`

Описывает **3 сервиса**:

| Сервис | Образ | Описание |
|---|---|---|
| `db` | `pgvector/pgvector:pg17` | PostgreSQL 17 с расширением pgvector. Volume для персистентности данных |
| `nginx` | `nginx:alpine` | Reverse proxy: проксирует 80 → Gunicorn. Отключена буферизация для SSE |
| `backend` | build из `./Dockerfile` | FastAPI-приложение. Зависит от `db`, запускается через Gunicorn |

```yaml
# Ключевые моменты:
services:
  db:
    image: pgvector/pgvector:pg17
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: ...
      POSTGRES_PASSWORD: ...
      POSTGRES_DB: ...

  backend:
    build: .
    depends_on:
      db:
        condition: service_healthy
    env_file: .env

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - backend
```

#### `nginx/nginx.conf`

```nginx
# Ключевые директивы:
location / {
    proxy_pass http://backend:8000;
    proxy_buffering off;          # критично для SSE
    proxy_cache off;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;
}
```

- `proxy_buffering off` — без этого Nginx буферизирует ответ и SSE-события приходят пачкой, а не потоком.

#### `Dockerfile`

```dockerfile
FROM python:3.14-slim
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main
COPY . .
CMD ["poetry", "run", "gunicorn", "-c", "gunicorn_conf.py", "src.main:app"]
```

#### `gunicorn_conf.py`

```python
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120          # увеличен из-за долгих LLM-запросов
keepalive = 5
accesslog = "-"
```

- `worker_class = "uvicorn.workers.UvicornWorker"` — Gunicorn управляет процессами, Uvicorn обрабатывает async-запросы внутри каждого воркера.
- `timeout = 120` — увеличен, т.к. цикл ARCADE (Analyst ↔ Critic) может занять 15–30 секунд.

#### `.env.example`

```env
ENV=dev

DB__HOST=db
DB__PORT=5432
DB__USER=postgres
DB__PASSWORD=changeme
DB__NAME=ii_lib
DB__ECHO=false
DB__POOL_SIZE=20
DB__MAX_OVERFLOW=10

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

LLM__ANALYST_MODEL=gpt-4o
LLM__CRITIC_MODEL=claude-sonnet-4-20250514
LLM__EMBEDDING_MODEL=text-embedding-3-small
LLM__EMBEDDING_DIM=1536
LLM__MAX_ARCADE_ITERATIONS=3
```

---

### `src/core` — Ядро приложения

#### `src/core/config.py`

Использует `pydantic-settings` для типобезопасной загрузки конфигурации из `.env`:

```python
class DatabaseSettings(BaseModel):
    host: str
    port: int
    user: str
    password: SecretStr
    name: str
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 10
    pool_recycle: int = 1800

    @computed_field
    def url(self) -> str:
        # postgresql+asyncpg://user:pass@host:port/dbname
        ...

class LLMSettings(BaseModel):
    analyst_model: str = "gpt-4o"
    critic_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    max_arcade_iterations: int = 3

class AppSettings(BaseSettings):
    env: Literal["dev", "stage", "prod"] = "prod"
    db: DatabaseSettings
    llm: LLMSettings
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr
```

- Вложенная структура через `env_nested_delimiter="__"` — переменная `DB__HOST` маппится в `settings.db.host`.
- LLM-ключи хранятся как `SecretStr`, чтобы не утечь в логи.

#### `src/core/exceptions.py`

Кастомные исключения, которые перехватываются через `@app.exception_handler`:

```python
class NotFoundError(Exception): ...
class ConflictError(Exception): ...
class DocumentParsingError(Exception): ...
class LLMProviderError(Exception): ...
```

В `main.py` регистрируются обработчики, которые превращают их в корректные HTTP-ответы (404, 409, 422, 502).

#### `src/core/logging.py`

Конфигурация Loguru: убирает стандартный handler, добавляет формат с timestamp и уровнем, опционально пишет в файл:

```python
from loguru import logger
import sys

def setup_logging(env: str) -> None:
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, format=fmt, level="DEBUG" if env == "dev" else "INFO")
```

---

### `src/db` — Слой базы данных

#### `src/db/base_model.py`

`DeclarativeBase` для всех SQLAlchemy-моделей с кастомным `__repr__`:

```python
class Base(DeclarativeBase):
    __abstract__ = True
    repr_cols_num = 3
    repr_cols = tuple()
    # Выводит первые N колонок в repr для удобной отладки
```

#### `src/db/base_dao.py`

Универсальный generic-DAO класс (`BaseDAO[T]`), от которого наследуются все модульные DAO:

| Метод | Описание |
|---|---|
| `find_one_or_none_by_id(id)` | Получение по PK |
| `find_all(filters, expressions, limit, offset, order_by)` | Гибкий поиск с пагинацией |
| `add(values, flush)` | Вставка одной записи |
| `add_many(instances, return_objects)` | Bulk-вставка |
| `update(filters, values)` | Обновление по фильтру |
| `delete(filters)` | Удаление по фильтру |

Все методы логируют ошибки через Loguru и оборачивают `SQLAlchemyError`.

#### `src/db/session.py`

```python
engine = create_async_engine(settings.db.url, pool_size=..., max_overflow=...)
async_session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
        await session.commit()   # auto-commit при успехе
        # rollback при исключении
```

---

### `src/modules/users` — Пользователи

Уже реализован. CRUD для управления профилями.

#### `users/models.py`

| Колонка | Тип | Описание |
|---|---|---|
| `id` | `int` PK | Автоинкремент |
| `name` | `String(255)` | Имя пользователя |
| `created_at` | `DateTime(tz)` | Время создания |
| `updated_at` | `DateTime(tz)` | Время обновления |

#### `users/schemas.py`

- `UserCreate` — валидация имени (strip, min 1, max 255)
- `UserUpdate` — partial update (все поля optional)
- `UserFilter` — фильтрация по имени
- `UserRead` — ответ клиенту (`from_attributes=True`)

#### `users/dao.py`

`UserDAO(BaseDAO[User])` — добавляет `find_filtered()` с case-insensitive partial match по имени.

#### `users/services.py`

`UserService` — тонкий слой бизнес-логики:
- `get_all()`, `get_by_id()`, `create()`, `update()`, `delete()`
- При `update()` вручную проставляет `updated_at`

#### `users/routers.py`

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/users/` | Список с фильтрацией и пагинацией |
| `GET` | `/users/{id}` | Один пользователь |
| `POST` | `/users/` | Создание (201) |
| `PATCH` | `/users/{id}` | Частичное обновление |
| `DELETE` | `/users/{id}` | Удаление (204) |

#### `users/dependencies.py`

FastAPI Depends: создаёт `UserService(UserDAO(session))`.

---

### `src/modules/documents` — Документы

Загрузка файлов, извлечение текста, разбиение на чанки.

#### `documents/models.py`

```python
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int]              # PK
    user_id: Mapped[int]         # FK → users.id
    filename: Mapped[str]        # Оригинальное имя файла
    content_type: Mapped[str]    # MIME-тип (application/pdf, text/plain, ...)
    file_path: Mapped[str]       # Путь к сохранённому файлу на диске / S3
    text_content: Mapped[str]    # Извлечённый текст (полный)
    status: Mapped[str]          # "pending" | "processing" | "ready" | "error"
    created_at: Mapped[datetime]
```

#### `documents/schemas.py`

```python
class DocumentUpload(BaseModel):
    user_id: int          # К какому пользователю привязан

class DocumentRead(BaseModel):
    id: int
    filename: str
    content_type: str
    status: str
    created_at: datetime
```

#### `documents/dao.py`

`DocumentDAO(BaseDAO[Document])` — специфичные запросы:
- `find_by_user(user_id)` — все документы пользователя
- `update_status(doc_id, status)` — обновление статуса обработки

#### `documents/services.py`

`DocumentService` — оркестрация полного пайплайна загрузки:

```
1. Сохранить файл на диск
2. Создать запись в БД (status="pending")
3. Вызвать parser.extract_text() → text_content
4. Вызвать chunking.split_into_chunks() → список чанков
5. Передать чанки в RagService для индексации
6. Обновить status="ready"
```

#### `documents/routers.py`

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/documents/upload` | Загрузка файла (`UploadFile`) |
| `GET` | `/documents/` | Список документов пользователя |
| `GET` | `/documents/{id}` | Метаданные документа |
| `DELETE` | `/documents/{id}` | Удаление документа и связанных чанков |

#### `documents/parser.py`

Асинхронное извлечение текста из файлов:

```python
async def extract_text(file_path: str, content_type: str) -> str:
    """
    Определяет тип файла и вызывает соответствующий парсер:
    - PDF → PyPDF2 / pdfplumber (в executor, т.к. sync)
    - TXT → простое чтение aiofiles
    - DOCX → python-docx (в executor)
    """
```

Sync-библиотеки оборачиваются в `asyncio.to_thread()` / `loop.run_in_executor()`.

#### `documents/chunking.py`

```python
async def split_into_chunks(
    text: str,
    doc_summary: str,    # краткое описание документа (от LLM)
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    1. Разбивает текст на куски по chunk_size токенов с перекрытием chunk_overlap
    2. Contextual Enrichment: к каждому куску в начало добавляется
       краткое описание всего документа, чтобы эмбеддинг чанка
       учитывал глобальный контекст
    """
```

---

### `src/modules/rag` — GraphRAG

Индексация и граф-поиск по документам. Вся граф-логика внутри PostgreSQL.

#### `rag/models.py`

Три таблицы:

```python
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int]                 # PK
    document_id: Mapped[int]        # FK → documents.id
    content: Mapped[str]            # Текст чанка
    chunk_index: Mapped[int]        # Порядковый номер в документе
    embedding: Mapped[Vector]       # VECTOR(1536) — pgvector
    created_at: Mapped[datetime]

class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int]                 # PK
    document_id: Mapped[int]        # FK → documents.id
    name: Mapped[str]               # Название сущности ("Пётр I", "RAG", ...)
    entity_type: Mapped[str]        # Тип: "person", "concept", "date", ...
    description: Mapped[str]        # Краткое описание (от LLM)

class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[int]                 # PK
    source_entity_id: Mapped[int]   # FK → entities.id
    target_entity_id: Mapped[int]   # FK → entities.id
    relation_type: Mapped[str]      # "authored", "mentions", "causes", ...
    description: Mapped[str]        # Описание связи
    chunk_id: Mapped[int]           # FK → document_chunks.id (где найдена связь)
```

#### `rag/schemas.py`

```python
class ChunkRead(BaseModel):
    id: int
    content: str
    chunk_index: int
    score: float | None = None       # cosine similarity при поиске

class EntityRead(BaseModel):
    id: int
    name: str
    entity_type: str
    description: str

class SearchResult(BaseModel):
    chunks: list[ChunkRead]
    entities: list[EntityRead]
    context_text: str                 # Собранный контекст для LLM
```

#### `rag/dao.py`

`RagDAO` — специализированные SQL-запросы:

```python
async def vector_search(self, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
    """SELECT *, embedding <=> :query AS score ... ORDER BY score LIMIT :top_k"""

async def graph_traversal(self, chunk_ids: list[int], hops: int = 2) -> SearchResult:
    """
    WITH RECURSIVE связанные AS (
        SELECT ... FROM relationships WHERE chunk_id IN :chunk_ids
        UNION ALL
        SELECT ... FROM relationships r JOIN связанные с ON ...
    )
    Рекурсивный CTE: от стартовых чанков по таблице relationships
    находит соседние сущности и их чанки за N хопов
    """
```

#### `rag/services.py`

`RagService` — оркестрация индексации и поиска:

```python
async def index_document(self, doc_id: int, chunks: list[str]) -> None:
    """
    1. Для каждого чанка вызвать embedder.embed() → вектор
    2. Сохранить DocumentChunk с вектором
    3. Вызвать graph_builder.extract_entities_and_relations() → сущности + связи
    4. Сохранить Entity и Relationship
    """

async def search(self, query: str, top_k: int = 5, hops: int = 2) -> SearchResult:
    """
    1. Embed запрос → вектор
    2. vector_search() → top_k ближайших чанков
    3. graph_traversal() → расширить контекст через граф связей
    4. Собрать итоговый SearchResult
    """
```

#### `rag/embedder.py`

```python
async def embed(text: str) -> list[float]:
    """
    Вызов OpenAI Embeddings API:
    POST https://api.openai.com/v1/embeddings
    model: text-embedding-3-small
    → вектор размерности 1536
    Используется httpx.AsyncClient для неблокирующего HTTP.
    """

async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Батчевая версия для эффективной индексации."""
```

#### `rag/graph_builder.py`

```python
async def extract_entities_and_relations(chunk: str) -> tuple[list[Entity], list[Relationship]]:
    """
    Отправляет чанк текста в LLM с промптом:
    "Извлеки сущности (имя, тип, описание) и связи между ними"
    Ответ валидируется через PydanticAI в строгую структуру.
    """
```

#### `rag/graph_search.py`

```python
async def multi_hop_search(
    dao: RagDAO,
    query_embedding: list[float],
    top_k: int = 5,
    hops: int = 2,
) -> str:
    """
    1. Находит top_k ближайших чанков по cosine similarity
    2. По chunk_id достаёт связанные сущности
    3. Через WITH RECURSIVE обходит граф на N хопов
    4. Собирает все упомянутые чанки и сущности
    5. Формирует текстовый контекст для LLM
    """
```

---

### `src/modules/agents` — Мультиагентная система

Реализация модели **ARCADE** через LangGraph.

#### `agents/state.py`

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str               # Вопрос пользователя
    context: str                # RAG-контекст из GraphRAG
    draft_answer: str           # Текущий черновик ответа (от Analyst)
    critique: str               # Замечания Critic
    is_approved: bool           # Принят ли ответ
    iteration: int              # Номер итерации цикла
    max_iterations: int         # Макс. число итераций (из конфига)
    final_answer: str           # Финальный ответ для пользователя
    events: list[dict]          # SSE-события для стриминга на фронт
```

#### `agents/tools.py`

Tool-функции, которые агенты могут вызывать:

```python
async def search_knowledge(query: str) -> str:
    """
    Инструмент для агента: поиск по базе знаний через GraphRAG.
    Вызывает RagService.search() и возвращает текстовый контекст.
    """
```

#### `agents/nodes/analyst.py`

```python
from pydantic_ai import Agent

analyst_agent = Agent(
    model="openai:gpt-4o",
    result_type=AnalystResponse,   # Pydantic-модель с полями answer, confidence
    system_prompt="Ты — аналитик. На основе контекста дай точный ответ...",
)

async def analyst_node(state: AgentState) -> AgentState:
    """
    1. Получает из state вопрос + контекст + (если есть) замечания критика
    2. Отправляет в LLM через PydanticAI
    3. Записывает draft_answer в state
    4. Добавляет SSE-событие {"type": "analyst_thinking", "data": ...}
    """
```

- **PydanticAI** гарантирует, что ответ LLM — валидный JSON, соответствующий `AnalystResponse`.

#### `agents/nodes/critic.py`

```python
critic_agent = Agent(
    model="anthropic:claude-sonnet-4-20250514",
    result_type=CriticResponse,    # {"is_approved": bool, "critique": str}
    system_prompt="Ты — критик. Проверь ответ на галлюцинации...",
)

async def critic_node(state: AgentState) -> AgentState:
    """
    1. Получает draft_answer и оригинальный контекст
    2. Сравнивает: все ли факты в ответе подтверждены контекстом
    3. Возвращает CriticResponse
    4. Добавляет SSE-событие {"type": "critic_review", "data": ...}
    """
```

- Намеренно используется **другая модель** (Anthropic вместо OpenAI), чтобы уменьшить вероятность одинаковых галлюцинаций.

#### `agents/workflow.py`

```python
from langgraph.graph import StateGraph, END

def build_workflow() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,            # is_approved=True → END, иначе → analyst
        {"continue": "analyst", "end": END},
    )

    return graph.compile()

def should_continue(state: AgentState) -> str:
    if state["is_approved"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    return "continue"
```

**Цикл ARCADE:**

```
Ask (пользователь) → Retrieve (GraphRAG) → Analyze (Analyst)
→ Critique (Critic) → Decide (условное ребро)
→ если не ок: обратно к Analyze с замечаниями
→ если ок: Emit (SSE → фронтенд)
```

---

### `src/modules/chat` — Чат и SSE-стриминг

#### `chat/models.py`

```python
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int]              # PK
    user_id: Mapped[int]         # FK → users.id
    title: Mapped[str | None]    # Автогенерируемый заголовок (по первому сообщению)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int]              # PK
    session_id: Mapped[int]      # FK → chat_sessions.id
    role: Mapped[str]            # "user" | "assistant" | "system"
    content: Mapped[str]         # Текст сообщения
    metadata_: Mapped[dict | None]  # JSON: итерации ARCADE, использованные чанки, и т.д.
    created_at: Mapped[datetime]
```

#### `chat/schemas.py`

```python
class ChatRequest(BaseModel):
    session_id: int | None = None   # None → создать новую сессию
    user_id: int
    message: str

class ChatEvent(BaseModel):
    """Формат SSE-событий, отправляемых на фронтенд"""
    event: str        # "token" | "analyst_thinking" | "critic_review" | "done" | "error"
    data: str         # Payload

class SessionRead(BaseModel):
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int

class MessageRead(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
```

#### `chat/dao.py`

`ChatDAO`:
- `create_session(user_id)` — создать новую сессию
- `add_message(session_id, role, content, metadata)` — сохранить сообщение
- `get_sessions_by_user(user_id)` — история сессий
- `get_messages_by_session(session_id)` — сообщения в сессии

#### `chat/services.py`

```python
class ChatService:
    async def stream_response(self, request: ChatRequest) -> AsyncGenerator[ChatEvent, None]:
        """
        1. Создать / получить сессию
        2. Сохранить сообщение пользователя
        3. Поиск контекста через RagService
        4. Запустить LangGraph workflow
        5. По мере работы yield ChatEvent (SSE):
           - {"event": "analyst_thinking", "data": "Генерирую ответ..."}
           - {"event": "token", "data": "Пётр"}
           - {"event": "critic_review", "data": "Проверяю факты..."}
           - {"event": "token", "data": " I правил..."}
           - {"event": "done", "data": ""}
        6. Сохранить финальный ответ в БД
        """
```

#### `chat/routers.py`

```python
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(request: ChatRequest, service: ChatService = Depends(...)):
    """SSE-стриминг ответа нейросети"""
    async def event_generator():
        async for event in service.stream_response(request):
            yield f"event: {event.event}\ndata: {event.data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/sessions", response_model=list[SessionRead])
async def get_sessions(user_id: int, ...): ...

@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def get_messages(session_id: int, ...): ...
```

- `X-Accel-Buffering: no` — дополнительный сигнал Nginx не буферизировать.

---

## API-эндпоинты

### Users

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `GET` | `/users/` | Список пользователей | ✅ Готов |
| `GET` | `/users/{id}` | Профиль пользователя | ✅ Готов |
| `POST` | `/users/` | Создать пользователя | ✅ Готов |
| `PATCH` | `/users/{id}` | Обновить профиль | ✅ Готов |
| `DELETE` | `/users/{id}` | Удалить пользователя | ✅ Готов |

### Documents

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `POST` | `/documents/upload` | Загрузить файл (PDF/TXT/DOCX) | ⬜ |
| `GET` | `/documents/` | Список документов пользователя | ⬜ |
| `GET` | `/documents/{id}` | Метаданные документа | ⬜ |
| `DELETE` | `/documents/{id}` | Удалить документ + чанки | ⬜ |

### Chat

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `POST` | `/chat/stream` | SSE-стриминг ответа | ⬜ |
| `GET` | `/chat/sessions` | Список сессий пользователя | ⬜ |
| `GET` | `/chat/sessions/{id}/messages` | История сообщений | ⬜ |

---

## Переменные окружения

| Переменная | Описание | Пример |
|---|---|---|
| `ENV` | Окружение | `dev` / `stage` / `prod` |
| `DB__HOST` | Хост PostgreSQL | `db` |
| `DB__PORT` | Порт PostgreSQL | `5432` |
| `DB__USER` | Пользователь БД | `postgres` |
| `DB__PASSWORD` | Пароль БД | `changeme` |
| `DB__NAME` | Имя базы данных | `ii_lib` |
| `DB__ECHO` | SQL-логирование | `false` |
| `DB__POOL_SIZE` | Размер пула соединений | `20` |
| `DB__MAX_OVERFLOW` | Макс. дополнительных соединений | `10` |
| `OPENAI_API_KEY` | Ключ OpenAI API | `sk-...` |
| `ANTHROPIC_API_KEY` | Ключ Anthropic API | `sk-ant-...` |
| `LLM__ANALYST_MODEL` | Модель для Analyst | `gpt-4o` |
| `LLM__CRITIC_MODEL` | Модель для Critic | `claude-sonnet-4-20250514` |
| `LLM__EMBEDDING_MODEL` | Модель для эмбеддингов | `text-embedding-3-small` |
| `LLM__EMBEDDING_DIM` | Размерность вектора | `1536` |
| `LLM__MAX_ARCADE_ITERATIONS` | Макс. итераций ARCADE | `3` |

---

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <repo-url> && cd II_lib_managment

# 2. Скопировать и заполнить .env
cp .env.example .env
# Заполнить ключи API и параметры БД

# 3. Запустить всё одной командой
docker-compose up --build -d

# 4. Применить миграции
docker-compose exec backend poetry run alembic upgrade head

# Приложение доступно на http://localhost
```

### Локальная разработка (без Docker)

```bash
# Установить зависимости
poetry install

# Запустить PostgreSQL с pgvector локально
# ...

# Применить миграции
poetry run alembic upgrade head

# Запустить dev-сервер
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```