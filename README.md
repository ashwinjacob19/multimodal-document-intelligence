# Multimodal Document Intelligence Platform

An enterprise AI platform that can ingest documents (PDFs, DOCX, images, URLs), understand text and visual content (tables, charts, diagrams), and answer grounded questions with citations using hybrid retrieval.

This is the foundation setup (Milestone 1) of the platform.

## Technology Stack

* **Python 3.12**
* **FastAPI** (Web framework)
* **uv** (Fast Python package manager)
* **PostgreSQL 16** with **pgvector**
* **SQLAlchemy 2.x** (Async ORM)
* **Alembic** (Database migrations)
* **Pydantic Settings** (Configuration management)
* **Ruff** (Linter/Formatter)
* **Docker** & **Docker Compose**

## Project Structure

```text
.
├── Dockerfile                  # Application Docker image configuration
├── README.md                   # Project documentation
├── alembic.ini                 # Alembic migrations config
├── app/                        # Application source directory
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings configuration manager
│   ├── db/
│   │   ├── __init__.py
│   │   └── session.py          # SQLAlchemy async session management
│   └── main.py                 # FastAPI application and endpoints
├── docker-compose.yml          # Local orchestration for app and db
├── migrations/                 # Alembic migration scripts and env
│   ├── env.py                  # Migration runner configuration
│   └── script.py.mako          # Migration script template
└── pyproject.toml              # Project dependencies and tool configurations
```

## Getting Started

### Prerequisites

* Docker and Docker Compose installed.

### Setup and Running

1. **Clone the repository and copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Start the application and database**:
   ```bash
   docker compose up --build
   ```
   This command builds the FastAPI container, starts the PostgreSQL database, waits until the database is ready, runs all database migrations, and spins up the FastAPI app on port `8000`.

### Verifying the Foundation

1. **Health Check Endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```
   Expected response:
   ```json
   {"status":"ok"}
   ```

2. **Readiness Check Endpoint** (verifies database connectivity):
   ```bash
   curl http://localhost:8000/ready
   ```
   Expected response (when database connection is successful):
   ```json
   {"status":"ready"}
   ```

## Development and Database Migrations

* **Create a new migration**:
  To auto-generate database migrations based on models, run:
  ```bash
  docker compose exec web alembic revision --autogenerate -m "description"
  ```

* **Run migrations manually**:
  ```bash
  docker compose exec web alembic upgrade head
  ```
