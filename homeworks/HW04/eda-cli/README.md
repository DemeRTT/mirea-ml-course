# HW04 – eda-cli: CLI + HTTP (FastAPI)

Проект **eda-cli**: небольшой инструмент для базового EDA CSV-файлов (HW03) и HTTP-сервис поверх него на **FastAPI** (HW04).

## Требования

- Python 3.11+
- установленный [uv](https://docs.astral.sh/uv/)

## Установка

В корне проекта:

```bash
uv sync
```

## CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports_example \
  --title "Example dataset report" \
  --max-hist-columns 8 \
  --top-k-categories 5 \
  --min-missing-share 0.15
```

Результаты в `reports_example/`:

- `report.md` – основной отчёт (Markdown)
- `summary.csv` – таблица по колонкам
- `missing.csv` – пропуски по колонкам
- `correlation.csv` – корреляционная матрица (если есть числовые признаки)
- `top_categories/*.csv` – top-k категорий по строковым признакам
- `hist_*.png` – гистограммы числовых колонок
- `missing_matrix.png` – визуализация пропусков
- `correlation_heatmap.png` – тепловая карта корреляций

## HTTP-сервис (FastAPI)

Запуск сервиса:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Документация Swagger:

- `http://localhost:8000/docs`

### Эндпоинты

- `GET /health` – health-check
- `POST /quality` – оценка качества по агрегированным признакам (JSON)
- `POST /quality-from-csv` – оценка качества по CSV (EDA-ядро)
- `POST /quality-flags-from-csv` – **полный набор булевых флагов качества по CSV** (включая новые эвристики HW03)

### Примеры запросов

Health:

```bash
curl -s http://localhost:8000/health
```

Оценка по CSV:

```bash
curl -s -F "file=@data/example.csv" http://localhost:8000/quality-from-csv
curl -s -F "file=@data/example.csv" http://localhost:8000/quality-flags-from-csv
```

## Тесты

```bash
uv run pytest -q
```
