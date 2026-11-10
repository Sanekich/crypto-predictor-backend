# CryptoPredictor Backend

Серверная часть CryptoPredictor: сбор рыночных данных, ML-модели и REST API (FastAPI).

> Прогнозы носят исследовательский характер и не являются финансовой рекомендацией.

## Требования

- Python 3.11+ (рекомендуется 3.12)
- Windows / PowerShell

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# PyTorch ставится отдельно (выберите сборку на https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cpu

Copy-Item .env.example .env
```

## Запуск

```powershell
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Health: http://127.0.0.1:8000/health
- Документация: http://127.0.0.1:8000/docs

## Тесты

```powershell
pytest
```

## Структура

```text
app/
  api/routes/   HTTP endpoints
  core/         настройки, логирование
  data/         сбор и обработка данных
  ml/           модели, обучение, оценка
  services/     бизнес-логика
  database/     слой хранения
tests/          автотесты
data/raw/       исходные данные (не в Git)
data/processed/ обработанные данные (не в Git)
artifacts/      обученные модели (не в Git)
```
