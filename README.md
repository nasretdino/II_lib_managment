# II Lib Management — NotebookLM-style Backend

> Учебный бэкенд для приложения-аналога NotebookLM: загрузка небольших документов, GraphRAG-поиск по ним (LightRAG) и чат с мультиагентной системой ARCADE на базе LangGraph.

**Это учебный проект.** Сюда прилетит пару маленьких файликов — никаких брокеров сообщений, Kubernetes, очередей задач и прочего оверинжиниринга. Всё должно работать просто, красиво и понятно.

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
| **Левая панель** — загрузка файлов | Приём PDF/TXT/DOCX (пара небольших файлов), парсинг через MarkItDown / PyMuPDF, разбиение на чанки, индексация через LightRAG |
| **Правая панель** — чат | SSE-стриминг ответа, мультиагентный ARCADE-пайплайн (6 узлов LangGraph) с валидацией через PydanticAI |
| **Профиль** | CRUD пользователей, история диалогов |

### Ключевые принципы

- **ARCADE** (Ask → Retrieve → Analyze → Critique → Decide → Emit) — вместо одного гигантского промпта задача разбита на 6 специализированных узлов LangGraph, каждый со своей ответственностью.
- **LightRAG** — библиотека для GraphRAG, которая берёт на себя построение графа знаний, извлечение сущностей/связей и гибридный поиск (vector + graph). Никакого ручного написания CTE-запросов и graph traversal.
- **Простота** — учебный проект без брокеров сообщений, очередей, Kubernetes. Docker Compose поднимает PostgreSQL + бэкенд + Nginx, и всё работает.
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
| База данных | PostgreSQL + pgvector |
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
                                    │  chat_messages                    │
                                    │  + LightRAG storage               │
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

### SSE и Nginx

Большинство обратных прокси, включая Nginx, по умолчанию **буферизируют** HTTP-ответы. Для SSE это убивает «typewriter effect» — пользователь не увидит ответа, пока LLM не сгенерирует его целиком.

**Решение:**

1. FastAPI отправляет заголовок `X-Accel-Buffering: no`
2. В конфигурации Nginx для SSE-маршрутов:

```nginx
location /chat/stream {
    proxy_pass http://backend:8000;

    proxy_buffering off;           # отключить буферизацию ответов
    proxy_cache off;               # не кэшировать
    chunked_transfer_encoding off; # SSE не нужен chunked
    proxy_http_version 1.1;
    proxy_set_header Connection '';

    proxy_read_timeout 300s;       # увеличен для долгих "размышлений" агента
}
```

3. **Keep-alive пинги**: для предотвращения разрыва соединения прокси-сервером во время долгих стадий (например, Critique), сервер транслирует пустые SSE-комментарии (`:\n\n`) каждые **15 секунд**.

---

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
        ├── users/               # ✅ Готов
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

---

## Описание каждого файла

### Инфраструктура (корень)

#### `docker-compose.yml`

Описывает **3 сервиса**:

| Сервис | Образ | Описание |
|---|---|---|
| `db` | `pgvector/pgvector:pg17` | PostgreSQL 17 с pgvector. Volume для персистентности |
| `nginx` | `nginx:alpine` | Reverse proxy с отключённой буферизацией для SSE |
| `backend` | build из `./Dockerfile` | FastAPI через Gunicorn |

```yaml
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
# Общий проксирование
location / {
    proxy_pass http://backend:8000;
    proxy_http_version 1.1;
}

# SSE-маршрут — отключение буферизации
location /chat/stream {
    proxy_pass http://backend:8000;
    proxy_buffering off;
    proxy_cache off;
    chunked_transfer_encoding off;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    proxy_read_timeout 300s;       # долгие "размышления" агентов
}
```

- `proxy_buffering off` — без этого SSE-события приходят пачкой, а не потоком.
- `proxy_read_timeout 300s` — на случай долгой стадии Critique.
- Сервер шлёт keep-alive пинги (`:\n\n`) каждые 15 секунд.

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

Конфигурация Loguru:

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
```

#### `src/db/base_dao.py`

Универсальный generic-DAO (`BaseDAO[T]`), от которого наследуются все модульные DAO:

| Метод | Описание |
|---|---|
| `find_one_or_none_by_id(id)` | Получение по PK |
| `find_all(filters, expressions, limit, offset, order_by)` | Гибкий поиск с пагинацией |
| `add(values, flush)` | Вставка одной записи |
| `add_many(instances, return_objects)` | Bulk-вставка |
| `update(filters, values)` | Обновление по фильтру |
| `delete(filters)` | Удаление по фильтру |

#### `src/db/session.py`

```python
engine = create_async_engine(settings.db.url, pool_size=..., max_overflow=...)
async_session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
        await session.commit()
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

Загрузка небольших файлов, извлечение текста, разбиение на чанки.

#### `documents/models.py`

```python
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int]              # PK
    user_id: Mapped[int]         # FK → users.id
    filename: Mapped[str]        # Оригинальное имя файла
    content_type: Mapped[str]    # MIME-тип (application/pdf, text/plain, ...)
    file_path: Mapped[str]       # Путь к сохранённому файлу на диске
    text_content: Mapped[str]    # Извлечённый текст (полный)
    status: Mapped[str]          # "pending" | "processing" | "ready" | "error"
    created_at: Mapped[datetime]
```

#### `documents/schemas.py`

```python
class DocumentUpload(BaseModel):
    user_id: int

class DocumentRead(BaseModel):
    id: int
    filename: str
    content_type: str
    status: str
    created_at: datetime
```

#### `documents/dao.py`

`DocumentDAO(BaseDAO[Document])`:
- `find_by_user(user_id)` — все документы пользователя
- `update_status(doc_id, status)` — обновление статуса обработки

#### `documents/services.py`

`DocumentService` — оркестрация пайплайна загрузки:

```
1. Сохранить файл на диск
2. Создать запись в БД (status="pending")
3. Вызвать parser.extract_text() → text_content
4. Вызвать chunking.split_into_chunks() → список чанков
5. Передать чанки в RagService для индексации через LightRAG
6. Обновить status="ready"
```

Так как проект учебный и файлы маленькие, весь пайплайн выполняется синхронно в рамках запроса — без очередей задач.

#### `documents/routers.py`

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/documents/upload` | Загрузка файла (`UploadFile`) |
| `GET` | `/documents/` | Список документов пользователя |
| `GET` | `/documents/{id}` | Метаданные документа |
| `DELETE` | `/documents/{id}` | Удаление документа и связанных данных |

#### `documents/parser.py`

Извлечение текста из файлов через **MarkItDown** или **PyMuPDF**:

```python
async def extract_text(file_path: str, content_type: str) -> str:
    """
    Определяет тип файла и извлекает текст:
    - PDF  → MarkItDown / PyMuPDF (в asyncio.to_thread, т.к. sync)
    - TXT  → простое чтение
    - DOCX → MarkItDown (в asyncio.to_thread)

    MarkItDown умеет конвертировать PDF, DOCX, PPTX и другие форматы
    в Markdown, сохраняя структуру документа (заголовки, списки, таблицы).
    PyMuPDF — альтернатива для PDF с надёжным извлечением текста.
    """
```

Sync-библиотеки оборачиваются в `asyncio.to_thread()`.

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

#### `agents/workflow.py`

```python
from langgraph.graph import StateGraph, END

def build_workflow() -> StateGraph:
    graph = StateGraph(AgentState)

    # 6 узлов ARCADE
    graph.add_node("ask", router_node)
    graph.add_node("retrieve", researcher_node)
    graph.add_node("analyze", analyst_node)
    graph.add_node("critique", critic_node)
    graph.add_node("decide", decision_node)
    graph.add_node("emit", emitter_node)

    # Прямой путь
    graph.set_entry_point("ask")
    graph.add_edge("ask", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "critique")
    graph.add_edge("critique", "decide")

    # Условное ребро: retry или финал
    graph.add_conditional_edges(
        "decide",
        should_continue,
        {
            "retry_retrieve": "retrieve",
            "retry_analyze": "analyze",
            "end": "emit",
        },
    )
    graph.add_edge("emit", END)

    return graph.compile()

def should_continue(state: AgentState) -> str:
    if state["is_approved"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    if state.get("needs_more_context"):
        return "retry_retrieve"
    return "retry_analyze"
```

---

### `src/modules/chat` — Чат и SSE-стриминг

#### `chat/models.py`

```python
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int]              # PK
    user_id: Mapped[int]         # FK → users.id
    title: Mapped[str | None]    # Автозаголовок (по первому сообщению)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int]              # PK
    session_id: Mapped[int]      # FK → chat_sessions.id
    role: Mapped[str]            # "user" | "assistant" | "system"
    content: Mapped[str]         # Текст сообщения
    metadata_: Mapped[dict | None]  # JSON: итерации ARCADE, источники
    created_at: Mapped[datetime]
```

#### `chat/schemas.py`

```python
class ChatRequest(BaseModel):
    session_id: int | None = None   # None → создать новую сессию
    user_id: int
    message: str

class ChatEvent(BaseModel):
    """Формат SSE-событий"""
    event: str    # "routing" | "retrieving" | "token" | "analyst_thinking" |
                  # "critic_review" | "done" | "error"
    data: str

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
        3. Запустить ARCADE workflow (LangGraph)
        4. По мере работы yield ChatEvent (SSE):
           - {"event": "routing", "data": "Определяю тип запроса..."}
           - {"event": "retrieving", "data": "Ищу в базе знаний..."}
           - {"event": "analyst_thinking", "data": "Генерирую ответ..."}
           - {"event": "token", "data": "Пётр"}
           - {"event": "critic_review", "data": "Проверяю факты..."}
           - {"event": "done", "data": ""}
        5. Keep-alive пинги (:\n\n) каждые 15 сек на стадии Critique
        6. Сохранить финальный ответ в БД
        """
```

#### `chat/routers.py`

```python
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(request: ChatRequest, service: ChatService = Depends(...)):
    async def event_generator():
        async for event in service.stream_response(request):
            yield f"event: {event.event}\ndata: {event.data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # сигнал Nginx не буферизировать
        },
    )

@router.get("/sessions", response_model=list[SessionRead])
async def get_sessions(user_id: int, ...): ...

@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def get_messages(session_id: int, ...): ...
```

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
| `DELETE` | `/documents/{id}` | Удалить документ | ⬜ |

### Chat

| Метод | Путь | Описание | Статус |
|---|---|---|---|
| `POST` | `/chat/stream` | SSE-стриминг ответа (ARCADE) | ⬜ |
| `GET` | `/chat/sessions` | Список сессий пользователя | ⬜ |
| `GET` | `/chat/sessions/{id}/messages` | История сообщений | ⬜ |

---

## Переменные окружения

| Переменная | Описание | Пример |
|---|---|---|
| `ENV` | Окружение | `dev` / `prod` |
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