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

1. 343 теста среды, генератора, схем, matcher, награды, атак и утечек;
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

Поведение адаптера покрыто тестами, которые повторяют сценарий жюри: `tests/test_judge_adapter.py`. Все примеры из опубликованных определений (тесты matcher 1–20 и M01–M12) воспроизведены в `tests/test_published_rules.py`.

**Рабочая директория и запуск в контейнере:** `/app`, `PYTHONPATH=/app`. Проверка: `python -m auditgym check`. Адаптер: `import judge_adapter`.

### Четыре типа ошибок

| ошибка из кейса | `type` | канонические `affected_ids` | сумма и валюта | минимальный набор доказательств |
|---|---|---|---|---|
| дубль счёта | `duplicate_invoice` | `invoice_id` оригинала и дубля | `total_minor` дубля, валюта счёта | `doc_id` обоих счетов |
| пропущенное или частичное распределение платежа | `payment_allocation` | `payment_id` и `invoice_id` | неразнесённая часть ссылки, валюта платежа | `doc_id` платежа, строка 2400 этого платежа, строки 1200/1950, которые применили платёж в его валюте |
| неверная дата или курс конвертации | `fx_rate` | `payment_id` и `invoice_id` | модуль разницы между проведённой суммой и пересчётом по policy, валюта счёта | `doc_id` платежа, строка 1950 в валюте платежа, строка 1200 (или 1950-дебет) в валюте счёта, `fx_id` каждого небазового курса |
| ошибка отнесения к периоду | `period_cutoff` | `invoice_id` | `total_minor` счёта, валюта счёта | `doc_id` счёта и строка 1200-дебет (или 4000-кредит) |

Правила matcher (код: `auditgym/judge/matcher.py`):

- ID, `type` и `currency` сравниваются точно, с учётом регистра и пробелов. Валюта пишется ровно так, как в `manifest.currency_exponents`: три заглавные буквы.
- `affected_ids` и `evidence_ids` сравниваются как множества.
- Допуск суммы τ = 0 для всех типов (`policy.amount_tolerance_minor`).
- Неизвестный тип, лишняя или пропущенная запись, неверная сумма или валюта дают FP.
- Структурная ошибка (−1) — это отсутствующее обязательное поле, неверный JSON-тип значения, больше 30 находок, больше 10 доказательств или объяснение длиннее 1000 символов. Лимиты проверяются до удаления повторов.
- Сопоставление — максимальное паросочетание. Если вариантов несколько, выбирается лексикографически наименьшая последовательность пар `(issue_id, индекс вывода)`.

### Формат эталона и provenance

`grader/truth.json`: `{"episode_id", "issues": [{"issue_id", "type", "affected_ids", "expected_amount_minor", "currency", "evidence_sets": [[...]], "scenario"}]}`. Поле `scenario` — `single`, `fx_partial_boundary` или `duplicate_fx`.

`grader/provenance.json`: `seed`, `config`, версии, `plan` и `mutations`. В каждой мутации есть `op`, `scenario`, затронутые ID, `ledger_rows` (изменённые или добавленные строки) и значения до и после. Как вернуть книги в состояние до ошибки:

| `op` | что сделано | как восстановить чистое состояние |
|---|---|---|
| `insert_duplicate_invoice` | добавлен документ-копия `duplicate` и его проводки `ledger_rows` | удалить документ `duplicate` и строки `ledger_rows` |
| `reduce_allocation` | применено `applied_after` вместо `applied_before`, разница ушла в 2400 | применить полную сумму ссылки из документа платежа, убрать её из 2400 |
| `wrong_rate_date` | пересчёт по курсу `used_rate_date` вместо `policy_rate_date` | заменить `converted_after` на `converted_before` в строках `ledger_rows` |
| `early_posting` | счёт, выставленный `issued_at` после конца периода, проведён `posted_at` внутри него | удалить строки `ledger_rows`: счёт относится к следующему периоду |

После такого восстановления все равенства из `DESIGN.md` («Проверка данных по файлам») выполняются. Это проверяет `python -m auditgym verify-episode`.

### Как получить каждый сценарий закрытых эпизодов

Конфигурация передаётся в `generate_episode(seed, config, output_dir)` напрямую или как `{"team_config": {...}}`. Ключи `errors` и `interactions` задают число ошибок каждого типа (число или `[min, max]`), остальные типы ставятся в 0:

| сценарий | config |
|---|---|
| D01 чистая компания | `{"preset": "clean"}` |
| D02 корректно разбитый платёж | `{"preset": "clean", "distractors": {"split_payments": true, "refunds": false, "recurring_amounts": false, "unreferenced_receipts": false, "timezone_edges": false}}` |
| D03 корректный возврат | то же, но `"refunds": true`, а остальные `false` |
| D04 законные повторяющиеся суммы | то же, но `"recurring_amounts": true` |
| D05 корректная валютная или периодная граница | то же, но `"timezone_edges": true`, плюс `"cross_currency_share": 0.5` |
| D06–D08 дубль счёта | `{"preset": "clean", "errors": {"duplicate_invoice": 1}}` |
| D09–D11 распределение платежа | `{"preset": "clean", "errors": {"payment_allocation": 1}, "allocation_mode": "missing"}` или `"partial"` |
| D12–D13 дата или курс конвертации | `{"preset": "clean", "errors": {"fx_rate": 1}}` |
| D14–D15 ошибка периода | `{"preset": "clean", "errors": {"period_cutoff": 1}}` |
| D16–D18 первое взаимодействие | `{"preset": "clean", "interactions": {"fx_partial_boundary": 1}}` — частичный платёж в другой валюте на границе периода |
| D19–D20 второе взаимодействие | `{"preset": "clean", "interactions": {"duplicate_fx": 1}}` — дубль счёта, оплата которого пересчитана неверным курсом |

Другие параметры (`period`, `customers`, `invoices`, `foreign_currencies`, `cross_currency_share`) меняют размер и состав компании. Все ключи и их допустимые значения описаны в `auditgym/configs/default.json` и проверяются в `auditgym/config.py`.

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
| `tests/` | 343 теста, включая сценарий жюри через `judge_adapter.py` и все опубликованные примеры matcher |
| `DESIGN.md` | записка о дизайне, поиск дыр, разбор сбоев, ограничения |
