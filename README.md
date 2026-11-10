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

## Загрузка рыночных данных

Свечи OHLCV загружаются с публичного API Binance (ключ не нужен):

```powershell
python scripts/download_data.py                        # BTCUSDT 1h, вся история с 2017-08-17
python scripts/download_data.py --interval 4h          # другой таймфрейм
python scripts/download_data.py --symbol ETHUSDT --start 2020-01-01
python scripts/download_data.py --full-refresh         # скачать заново, игнорируя сохранённое
```

Повторный запуск докачивает только новые свечи. Результат:

| Файл | Содержимое |
|---|---|
| `data/raw/binance/BTCUSDT_1h.parquet` | данные как пришли от Binance |
| `data/processed/binance/BTCUSDT_1h.parquet` | только закрытые свечи, UTC, без дубликатов, проверенные |
| `data/processed/binance/BTCUSDT_1h.report.json` | отчёт о качестве: ошибки и пропуски |

Если проверка находит ошибки (некорректные цены, дубликаты и т. п.), processed-файл
не обновляется. Пропуски свечей (например, техработы биржи) не заполняются
автоматически — они перечислены в отчёте.

## Признаки для ML

После загрузки данных собрать датасет (признаки + целевая доходность):

```powershell
python scripts/build_features.py                       # BTCUSDT 1h, горизонт 24h
python scripts/build_features.py --horizon 4h
```

Результат: `data/features/binance/BTCUSDT_1h_h24h.parquet` и `.meta.json`
(список и порядок признаков, параметры, период, контрольная сумма исходного файла).

- Признаки безразмерные (доходности, отношения), без абсолютных цен.
- Свечи раскладываются по равномерной сетке: окна, задевающие пропуск, дают NaN,
  такие строки исключаются.
- Target: `target_return_24h = close[t+24] / close[t] - 1` и направление (рост / не рост).
- Масштабирование (scaler) здесь не делается — только при обучении, на train-части.

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
