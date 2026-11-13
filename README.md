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

## Подготовка данных и моделей (на каждом новом компьютере)

Данные (`data/`) и обученные модели (`artifacts/`) не хранятся в Git. После клонирования
и установки зависимостей их нужно создать одной командой (~3–5 минут):

```powershell
.\.venv\Scripts\python.exe scripts\setup.py
```

Скрипт скачает свечи с Binance, соберёт признаки и обучит модели диапазона на 24h и 7d.
Повторный запуск только докачивает новые данные; `--retrain` обучает модели заново.
Без этого шага API отвечает ошибкой `MODEL_NOT_AVAILABLE`.

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

## Сбор живых данных

Стакан заявок, поток сделок, funding, open interest и ликвидации нельзя скачать задним
числом — их нужно записывать в момент, когда они происходят:

```powershell
.\.venv\Scripts\python.exe scripts\collect_live.py     # работает, пока не нажать Ctrl+C
```

- Данные: `data/live/<набор>/BTCUSDT/<дата>/<час>.parquet`, одна строка в минуту.
- Наборы: `trades_spot`, `trades_futures`, `book_spot`, `book_futures`, `mark_price`,
  `open_interest`, `liquidations`, `connection_events` (подключения/обрывы — для учёта пропусков).
- Состояние: `GET /live/status/BTCUSDT` или файл `data/live/_status_BTCUSDT.json`.
- Обрывы связи переподключаются автоматически; неполные минуты не записываются.
- Запускайте **один** экземпляр сборщика. Каждый час простоя — пропуск в истории.

План использования этих данных: [docs/direction_prediction_plan.md](docs/direction_prediction_plan.md).

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

## Обучение моделей

```powershell
python scripts/train_model.py                          # BTCUSDT 1h, горизонт 24h
python scripts/train_model.py --no-save                # эксперимент без сохранения
```

Обучаются и сравниваются на одном и том же разбиении:

- `zero_return` — цена не изменится (доходность 0);
- `historical_mean` — средняя доходность train-периода;
- `ridge` — линейная модель (scaler обучается только на train);
- `xgboost` — градиентный бустинг, число деревьев выбирается early stopping по validation.

Разбиение хронологическое (70% / 15% / 15%), перед validation и test удаляется
зазор длиной в горизонт прогноза, чтобы целевые окна не пересекались.
Модель выбирается только по validation; test — только для итогового отчёта.

Модель сохраняется в `artifacts/models/<model_id>/`: веса (`model.json`),
параметры и `metadata.json` (признаки и их порядок, версия pipeline, периоды,
метрики, сравнение с baseline, важность признаков, commit кода, хеш датасета).

### Модель диапазона (используется API)

```powershell
python scripts/build_features.py --horizon 7d
python scripts/train_range_model.py --horizon 24h
python scripts/train_range_model.py --horizon 7d
```

XGBoost с квантильной функцией потерь предсказывает 10-й, 50-й и 90-й процентили
будущей доходности, т.е. диапазон с вероятностью 80%. Сравнивается с постоянным
историческим диапазоном по pinball loss, покрытию и ширине.

### Модели направления на 1–4 часа (сигналы)

```powershell
.\.venv\Scripts\python.exe scripts\train_direction_model.py --horizon 1h
.\.venv\Scripts\python.exe scripts\train_direction_model.py --horizon 4h
```

Классификатор XGBoost с калибровкой вероятностей (Platt) на отдельном периоде.
Модель получает статус `approved`, только если её сигналы с уверенностью ≥ 75% прошли
проверку на тесте (≥ 77 независимых сигналов, точность ≥ 75%, нижняя граница ≥ 70%);
иначе — `shadow` (сигналы видны, но помечены как непроверенные).
Подробно: [docs/direction_prediction_plan.md](docs/direction_prediction_plan.md).

### Направление на день и неделю

Для прогноза направления на тех же горизонтах, что и диапазон цены, обучаются отдельные
калиброванные direction-модели:

```powershell
python scripts/train_direction_model.py --horizon 24h
python scripts/train_direction_model.py --horizon 7d
```

API использует direction-модель, соответствующую запрошенному горизонту.

**Правило показа направления.** При обучении модель проверяется на 5 последовательных
периодах (walk-forward). Направление показывается на сайте, только если модель лучше
базовой частоты роста (Brier skill > 0) в среднем и минимум в 4 из 5 периодов
(`walk_forward.has_skill` в метаданных). Иначе API отвечает `NO_SIGNAL`.

Результат на 12.11.2026: 1h и 4h — слабый, но стабильный сигнал (+0,9% Brier skill,
5 из 5 периодов); 24h — сигнала нет (2/5); 7d — модель хуже базовой частоты (0/5).
Ручное «сжатие» вероятностей 7d (0.35) удалено: walk-forward показал отрицательный
результат во всех периодах с ним и без него.

### Проверка стабильности направления

Для оценки direction-модели на нескольких последовательных отложенных периодах:

```powershell
python scripts/walk_forward_report.py --horizon 24h
python scripts/walk_forward_report.py --horizon 7d
```

Команда обучает временные fold-модели, ничего не сохраняет и не заменяет production-артефакты.
Также в `app/features/targets.py` добавлена опциональная разметка `down / neutral / up` для
будущего multiclass-обучения; текущие бинарные модели от неё не меняются.

Экспериментальный trainer для этой разметки:

```powershell
python scripts/train_neutral_direction_model.py --horizon 24h --no-save
python scripts/train_neutral_direction_model.py --horizon 7d --no-save
python scripts/train_neutral_direction_model.py --horizon 7d --neutral-return 0.01 --no-save
```

Для 7d эксперимент с `--neutral-return 0.01` означает: доходность выше +1% — `up`, ниже
-1% — `down`, между ними — `neutral`. Trainer использует inverse-frequency class weights
и выводит balanced accuracy, macro-F1 и confusion matrix, чтобы модель не игнорировала
neutral-класс. Он не заменяет текущие модели и не подключён к API, пока качество всех
трёх классов не будет подтверждено на независимых walk-forward-периодах.

### Этапы 2–3: признаки деривативов (история)

Funding rate, basis (premium index), open interest и позиции трейдеров, индекс
волатильности DVOL скачиваются за годы назад — ждать сборщика не нужно:

```powershell
python scripts/download_derivatives.py          # первый запуск ~15–30 мин, затем докачка
python scripts/compare_feature_sets.py          # walk-forward: базовые признаки против +деривативы
```

Все наборы сравниваются на одних и тех же строках и периодах. Набор считается улучшением
(`BETTER`) только если он лучше базового в среднем и в большинстве периодов. Признаки
используют только данные, опубликованные до закрытия свечи (`available_at`). Модели с
этими признаками пока только для офлайн-сравнения: API не считает их при прогнозе.

Результат на 12.11.2026 (5 периодов, 2021-03 … 2026-11, отчёт
`data/reports/feature_comparison_20261112_2048.json`):

| Задача | Базовые свечи | Лучший набор с деривативами | Вывод |
|---|---|---|---|
| направление 4h (Brier skill) | +0.0021 | +0.0031, все признаки | прирост мал, лучше в 3 из 5 периодов: неясно |
| направление 24h (Brier skill) | −0.0170 | −0.0130, +DVOL | меньше минус, но умения всё равно нет |
| диапазон 24h (pinball skill) | +3.2% | +3.0% | не лучше |
| диапазон 7d (pinball skill) | −0.9% | −0.7%, funding+basis | меньше минус, но умения всё равно нет |

Часовые funding, basis, OI и DVOL заметно прогноз не улучшают, поэтому рабочие модели
остаются на свечных признаках. Сигналов ≥ 75% нет ни в одном наборе: максимальная
уверенность около 65%.

### Новая логика: направление заметного движения (`artifacts/barrier`)

Старые модели отвечают на вопрос «цена через N часов будет выше или ниже?». Большая часть
таких движений — шум в доли процента, поэтому их уверенность упирается в 60–65%.

Новая логика спрашивает: **«что наступит раньше — рост на X% или падение на X%?»**, ожидая
не дольше заданного срока. Пороги симметричны: 75% попаданий означают, что направление
действительно угадано в 3 случаях из 4, а не подобрано асимметричной целью. Конфигурации
выбраны по истории BTCUSDT 1h так, чтобы почти всегда один из барьеров достигался:

| Конфигурация | Разрешается (барьер достигнут) | Медиана времени до пробоя |
|---|---|---|
| `move_1pct_24h` — ±1% в течение 24 ч | 91% | 4 ч |
| `move_2pct_72h` — ±2% в течение 3 дней | 94% | 12 ч |
| `move_3pct_168h` — ±3% в течение 7 дней | 96% | 24 ч |

Каждая конфигурация обучается на двух наборах данных: `candles` (только свечи, история с 2017)
и `all_data` (свечи + funding + basis + open interest + DVOL, история с 2021-03):

```powershell
python scripts/download_derivatives.py                 # свежая история деривативов
python scripts/train_barrier_models.py                 # 3 конфигурации x 2 набора данных
python scripts/train_barrier_models.py --config move_1pct_24h --feature-set candles
```

Модели сохраняются в `artifacts/barrier/models/<model_id>/` — отдельно от старой логики в
`artifacts/models` — и пока не используются в прогнозах на сайте. Проверка: обычный тестовый
период плюс 5 периодов walk-forward; уверенные сигналы (≥ 75%) со всех тестовых периодов
суммируются. Цель та же, что для старой логики: не меньше 77 независимых разрешившихся
сигналов, ≥ 75% попаданий, нижняя граница 95% ≥ 70% — при положительном Brier skill
в 4 из 5 периодов. Статус `approved` присваивается только при достижении цели.

API: `GET /barrier/overview`, `GET /barrier/models/{id}`, `DELETE /barrier/models/{id}` (токен).
Обучение из интерфейса: операция `train_barrier` (страница «Новая логика» во фронтенде).

### Качество по рыночным режимам

Чтобы проверить, где direction-модель работает лучше или хуже:

```powershell
python scripts/regime_report.py --horizon 24h
python scripts/regime_report.py --horizon 7d
```

Отчёт разделяет тестовый период на тренд вверх, тренд вниз, боковое движение и высокую
волатильность. Режимы определяются только по прошлым данным. Автоматического отключения
сигналов по режиму пока нет — сначала нужно накопить статистику.

## API

| Метод | Адрес | Что возвращает |
|---|---|---|
| GET | `/health` | состояние сервера |
| GET | `/assets` | активы и горизонты, для которых есть модели |
| GET | `/predictions/{symbol}?horizon=24h` | диапазон цены, медиана, качество модели |
| GET | `/signals/{symbol}?min_confidence=0.75` | направление на 1–4 ч, сигналы ≥ 75% и их журнал |
| GET | `/live/status/{symbol}` | состояние сборщика живых данных |
| GET | `/models` | список сохранённых моделей |
| GET | `/models/{model_id}` | все метаданные модели |
| GET | `/barrier/overview` | новая логика: конфигурации, критерии цели и прогресс моделей |
| GET | `/barrier/models/{model_id}` | все метаданные модели новой логики |

При запросе прогноза сервер сам докачивает новые закрытые свечи (не чаще раза в минуту).
Если данные устарели или модели нет, возвращается ошибка вида
`{"detail": {"code": "DATA_STALE", "message": "..."}}`, а не выдуманный прогноз.

Фронтенду разрешены запросы с адресов из `CORS_ORIGINS` (по умолчанию Vite dev server, порт 5173).

## Хостинг на Render

В репозитории есть `render.yaml`. Render → **New → Blueprint** → выбрать этот репозиторий.

- Web Service, регион **Frankfurt** (из США Binance недоступен), план Free.
- Данные и модели берутся из репозитория, переобучение не нужно.
- CORS разрешён для фронтенда на `https://crypto-predictor-frontend*.onrender.com`.

Ограничения бесплатного плана: сервис засыпает после 15 минут без запросов (первый
запрос после этого ~30–60 с), диск временный — журнал сигналов и докачанные свечи
сбрасываются при перезапуске. Сборщик живых данных на Free не работает: нужен
Background Worker и Persistent Disk (платные).

## Тесты

```powershell
pytest
```

## Быстрая проверка состояния данных и моделей

Для read-only проверки исторических данных, live-сборщика и статусов сохранённых
моделей выполните из корня репозитория:

```powershell
python scripts/status_report.py
```

Команда показывает количество свечей, период данных, пропуски, свежесть live-статуса,
ошибки парсинга и статусы моделей (`shadow`/`approved`). Она не обучает модели и не
изменяет файлы.

## Автоматический live-сбор и обучение

Для запуска API вместе с live-сборщиком:

```powershell
python scripts/start_backend_with_collector.py
```

В Render этот supervisor запускается автоматически через `render.yaml`. Collector
перезапускается, если процесс завершился. Для сохранения данных между перезапусками
нужен постоянный диск или внешнее хранилище; ephemeral-диск Render нельзя считать
долговременной обучающей выборкой.

Проверка покрытия данных:

```powershell
python scripts/status_report.py
```

Обучение direction-моделей прямо сейчас:

```powershell
python scripts/train_direction_model.py --horizon 24h
python scripts/train_direction_model.py --horizon 7d
```

После обучения проверь `accuracy`, `Brier score`, количество независимых сигналов,
`hit rate` и статус `shadow`. Не заменяй старую модель без walk-forward и paper/shadow
проверки на новых данных.

### Управление из frontend

Вкладка `#/tests` запускает только allow-listed операции: проверку данных, обучение
direction-модели, walk-forward и regime-report. Для защиты тяжёлых процессов задай на
backend секретную переменную `OPERATIONS_TOKEN`; этот же токен вводится в форме вкладки
на время сессии и не сохраняется в браузере. Одновременно разрешён только один запуск.

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
