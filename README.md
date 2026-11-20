# auditgym — RL-среда для ИИ-аудитора дебиторской задолженности

Воспроизводимый симулятор: генератор вымышленных компаний со скрытыми ошибками, среда в стиле Gymnasium с узкими инструментами расследования, оценщик в отдельном процессе, общий matcher и терминальная награда, JSON-схемы, базовые политики и записанные траектории.

Все компании, люди, счета и деньги вымышлены. Сеть, GPU и платные API не нужны. Зависимостей во время работы нет, для тестов нужен только pytest.

## Единая команда проверки

```bash
docker build -t auditgym .
docker run --rm --network none --cpus 2 --memory 4g auditgym
```

или без Docker (Python 3.11+):

```bash
pip install -r requirements-dev.txt
python -m auditgym check
```

Что делает `check` (около 15 секунд):

1. 290 тестов среды, генератора, схем, matcher, награды, атак и утечек;
2. генерирует 240 эпизодов всех пресетов и для каждого проверяет равенства, эталон, доказательства и решаемость в пределах 100 шагов;
3. экспортирует 10 train + 10 dev эпизодов в `episodes/`;
4. независимо проверяет экспортированные файлы, затем их копии с перемешанными строками и новыми ID;
5. прогоняет базовые политики и пишет `runs/<split>/trajectories.jsonl`, `runs/<split>/judge_log.jsonl` и `reports/metrics.md`. Train-сплит идёт в открытом режиме `train`: терминальная награда видна в траекториях. Dev-сплит идёт в закрытом режиме `eval`: награда в траекториях всегда 0 и хранится только в журнале оценщика.

Последняя строка вывода — `CHECK PASSED`.

## Интерфейс для жюри: `judge_adapter.py`

Файл лежит в корне репозитория (в контейнере это `/app/judge_adapter.py`, `PYTHONPATH=/app`) и реализует опубликованный контракт проверок:

```python
from judge_adapter import generate_episode, make_env, score_episode

generate_episode(seed, config, output_dir)
env = make_env()
obs, info = env.reset(seed, {"mode": "closed", "episode_dir": output_dir, "team_config": {}})
obs, reward, terminated, truncated, info = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
score = score_episode(output_dir, findings, inspection_steps)
```

- **`generate_episode(seed, config, output_dir)`** пишет эпизод в `output_dir/public/` (manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv) и в `output_dir/grader/` (truth.json, provenance.json). `config` — наша конфигурация (например `{"preset": "heavy"}`) или обёртка `{"team_config": {...}}`.
- **`make_env()`** возвращает среду. В закрытом режиме (`"mode": "closed"` и `episode_dir`) она читает только `public/` и никогда не открывает `grader/`, награда всегда 0, `info` пустой, а `submit` отвечает только `{"accepted": true}` или ограниченной ошибкой структуры. С `"mode": "train"` терминальная награда видна.
- **`score_episode(episode_dir, findings, inspection_steps)`** возвращает `{tp, fp, fn, precision, recall, f1, return}`. Неверная структура даёт `return = -1`, остальные поля `null`, а ошибка пишется в журнал оценщика (`AUDITGYM_JUDGE_LOG`, по умолчанию `judge_log.jsonl`).
- Действие — `{"name": ..., "args": {...}}`. Первое наблюдение — `{task, manifest, policy, remaining_steps: 100}`, следующие — `{result, error, remaining_steps}`.
- Сравнение ID точное, `affected_ids` и `evidence_ids` сравниваются как множества, допуск по сумме 0.

Поведение адаптера покрыто тестами, которые повторяют сценарий жюри: `tests/test_judge_adapter.py`.

## Как проверить каждый критерий

| критерий | где смотреть | команда |
|---|---|---|
| Контракт и изоляция | `auditgym/env.py`, `auditgym/judge/`, `tests/test_env_contract.py`, `tests/test_isolation.py`, `tests/test_hardening.py` | `python -m pytest -q tests/test_env_contract.py tests/test_isolation.py tests/test_hardening.py` |
| Данные и эталон | `auditgym/generator/`, `auditgym/verify.py`, `auditgym/episode_check.py` | `python -m auditgym verify-episode episodes/train episodes/dev` |
| Надёжность награды | `auditgym/judge/matcher.py`, `tests/test_matcher_reward.py`, `tests/test_reward_end_to_end.py` | `python -m pytest -q tests/test_matcher_reward.py tests/test_reward_end_to_end.py` |
| Воспроизводимость и обобщение | `auditgym/transform.py`, `tests/test_reproducibility.py`, `tests/test_files_and_transforms.py` | `python -m auditgym transform episodes/dev/<id> /tmp/moved --seed 5` |
| Инженерия и объяснение | `README.md`, `DESIGN.md`, `auditgym/configs/`, `runs/`, `reports/` | `python -m auditgym check` |

`verify-episode` работает только с файлами эпизода и для каждого проверяет:

- схемы;
- число записей;
- уникальность ID и целостность ссылок;
- бухгалтерские равенства;
- что каждое эталонное расхождение доказывается видимыми записями;
- что полная сверка по политике находит ровно объявленные расхождения, без лишних и без пропусков;
- что provenance совпадает с эталоном.

## Подключение своих эпизодов и преобразований

Среда загружает эпизод из папки так же, как генерирует по seed:

```python
env.reset(0, {"episode_dir": "episodes/dev/EP-..."})
```

Чтобы получить копию с перемешанными строками всех таблиц и новыми ID (документы, счета, платежи, проводки, контрагенты, episode_id), согласованно изменёнными в `public/`, `truth.json` и `provenance.json`:

```bash
python -m auditgym transform episodes/dev/EP-... /tmp/moved --seed 5
```

Перед загрузкой оценщик проверяет схемы и достаточность доказательств. Испорченные или несогласованные файлы отклоняются.

## Использование из Python

```python
from auditgym.env import AuditEnv

env = AuditEnv(mode="train")
obs, info = env.reset(seed=42, config={"preset": "interacting"})
episode_id = info["episode_id"]
obs, reward, terminated, truncated, info = env.step(
    {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}}
)
obs, reward, terminated, truncated, info = env.step(
    {"name": "submit", "args": {"findings": {"episode_id": episode_id, "findings": []}}}
)
env.close()
```

`AuditEnv(mode="train")` запускает оценщика отдельным процессом. `reset` также принимает `options=` вместо `config=` и `seed=None`, как в Gymnasium.

Действие — JSON-объект `{"name": ..., "args": {...}}`, можно строкой JSON:

| действие | аргументы |
|---|---|
| `list_records` | `table` ∈ ledger, invoices, payments, fx; `cursor` (null или из прошлой страницы); `limit` 1..50 |
| `get_record` | `table`, `id` (row_id, invoice_id, payment_id или fx_id) |
| `get_document` | `doc_id` |
| `calculator` | `expression`: только числа, `+ - * /`, скобки |
| `submit` | `findings`: `{episode_id, findings: [...]}` |

Режимы:

- **`train`** возвращает терминальную награду и метрики в `info`.
- **`closed`** (он же `eval`) всегда отдаёт награду 0 и пустой `info`. Результаты остаются у оценщика: `score_episode`, `JudgeClient.results()` и `--log`.

Ответ на действие — `{"result": ..., "error": null | {"code", "message"}, "remaining_steps": n}`: `list_records` даёт `{"records", "next_cursor"}`, `get_record` и `get_document` дают сам объект, `calculator` даёт десятичную строку.

Для агента в отдельном процессе есть `python -m auditgym serve --mode eval`: одна JSON-строка на запрос (`reset`, `step`, `close`) через stdin/stdout.

## Данные эпизода

```
episodes/<split>/<episode_id>/
  public/   manifest.json  policy.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv
  grader/   truth.json  provenance.json
```

Схемы всех форматов лежат в `auditgym/schemas/` (JSON Schema 2020-12): manifest, policy, ledger_row, invoice, payment, fx_row, findings, truth, action, observation, trajectory_record.

- **Деньги** — целые минимальные единицы, разрядность указана в `manifest.currency_exponents`.
- **Время** — ISO-8601 со смещением.
- **Курсы** — десятичные строки: `1 source = rate base`.
- **Отсутствующие ссылки** — явный `null` (в CSV — `null`).

Правила периода, выбора даты курса, проводок и формат каждого типа вывода лежат в `policy.json` и показываются агенту в первом наблюдении.

## Конфигурации

Версионированные пресеты: `auditgym/configs/*.json` (`config_version` = 1.0) — `clean`, `standard`, `single`, `interacting`, `heavy`, `quarter`. Любой ключ можно переопределить:

```python
env.reset(7, {"preset": "standard",
              "errors": {"duplicate_invoice": 2, "payment_allocation": 0, "fx_rate": [1, 3], "period_cutoff": 1},
              "interactions": {"fx_partial_boundary": 1, "duplicate_fx": 0},
              "presentation_seed": 99, "ledger_order": "shuffled"})
```

`presentation_seed` и `ledger_order` меняют только порядок и ID: содержание, эталон и награда oracle сохраняются.

Для закрытого прогона оценщик берёт секрет из переменной `AUDITGYM_SECRET` и подмешивает его в генерацию. Так по открытому коду и маленькому seed эпизод нельзя восстановить перебором.

## Все команды

```bash
python -m auditgym check                                   # полная проверка
python -m auditgym demo --seed 2004 --preset interacting   # скриптовый аудитор по шагам
python -m auditgym export --out episodes                   # 10 train + 10 dev эпизодов
python -m auditgym eval --split train,dev                  # траектории, журнал оценщика, метрики
python -m auditgym validate --seeds 0-99                   # самопроверка генератора
python -m auditgym verify-episode episodes/dev             # независимая проверка файлов эпизодов
python -m auditgym transform SRC DST --seed 5              # перестановка строк и смена ID
python -m auditgym generate --seed 7 --preset heavy        # один эпизод в файлы
python -m auditgym serve --mode eval                       # среда через stdin/stdout
```

## Состав репозитория

| путь | что внутри |
|---|---|
| `judge_adapter.py` | интерфейс для жюри: `generate_episode`, `make_env`, `score_episode` |
| `auditgym/generator/` | мир компании, инжекторы ошибок, рендер в открытые таблицы |
| `auditgym/verify.py` | проверка эталона только по видимым доказательствам |
| `auditgym/reconcile.py` | полная сверка по опубликованной политике |
| `auditgym/episode_check.py` | независимая проверка эпизода по файлам |
| `auditgym/transform.py` | безвредные преобразования эпизода и загрузка из файлов |
| `auditgym/judge/` | оценщик: matcher, награда, процесс-сервер и клиент |
| `auditgym/env.py` | среда reset/step |
| `auditgym/schemas/` | JSON-схемы всех форматов |
| `auditgym/configs/` | версионированные пресеты |
| `auditgym/policies/` | случайный исследователь, наивная сверка, «пометить всё», oracle, скриптовый аудитор |
| `episodes/` | 10 train + 10 dev эпизодов |
| `runs/<split>/` | `trajectories.jsonl` всех базовых политик и `judge_log.jsonl` |
| `reports/` | метрики и отчёт самопроверки |
| `tests/` | 290 тестов, включая сценарий жюри через `judge_adapter.py` |
| `DESIGN.md` | записка о дизайне, поиск дыр, разбор сбоев, ограничения |
