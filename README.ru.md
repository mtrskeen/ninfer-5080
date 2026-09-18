# NInfer — форк для RTX 5080 16 ГБ

**Qwen3.8-27B с контекстом 124 928 токенов на одной потребительской 16-гигабайтной Blackwell-карте.**

Форк развивает [aljazceru/ninfer](https://github.com/aljazceru/ninfer), который,
в свою очередь, основан на [Neroued/ninfer](https://github.com/Neroued/ninfer).
Добавлено: порт groupwise-int путей под Blackwell (`sm_120a`), 16-гигабайтный
артефакт Qwen3.8-27B (min-Q4 текст + **MTP-Q4** + **Vision-Q4**), быстрый
prefill-профиль и воспроизводимая кампания проверки.

- Артефакт: [`mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080`](https://huggingface.co/mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080) (15.5 ГБ, только для NInfer)
- Подробные результаты: [docs/BENCHMARKS.ru.md](docs/BENCHMARKS.ru.md)
- Рецепт сборки и квантования с обоснованием: [docs/BUILDING.ru.md](docs/BUILDING.ru.md)
- Сырые замеры: [validation/5080-20260918](validation/5080-20260918)
- English version: [README.md](README.md)

## Что умеет

- Одиночный текстовый запрос до 124 928 токенов со спекулятивным MTP3.
- Vision (изображения/видео) отдельным режимом `--vision`, потолок 32 768 токенов.
- Prefill в 2.35–2.53× быстрее прежнего дефолта за счёт `--prefill-chunk 256`.
- Вся модель плюс KV-пул INT4 на 124 928 токенов помещаются в 15.9 ГБ VRAM.

## Рекомендуемый запуск (RTX 5080 16 ГБ)

```bash
ninfer-serve qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --model-id qwen3.8-27b-16gb \
  --host 0.0.0.0 --port 18100 \
  --max-context 124928 --kv-capacity 124928 \
  --kv-dtype i4 --spec mtp --draft-tokens 3 \
  --prefill-chunk 256 --no-cuda-graph
```

Режим Vision (отдельный сервер): добавить `--vision` и выставить
`--max-context 32768 --kv-capacity 32768`.

## Результаты кратко

Замеры одиночные, один постоянный сервер, переиспользование префикса выключено.
Это **разные рантаймы и GPU**, таблица — только для ориентира; методика и сырые
данные — в [docs/BENCHMARKS.ru.md](docs/BENCHMARKS.ru.md).

| Вариант | Железо | Рантайм | Контекст | decode на длинной генерации | Примечания |
|---|---|---|---:|---:|---|
| Базовый llama.cpp | RTX 5080 16 ГБ (тот же хост) | llama.cpp b10853, UD-IQ3_XXS + q8_0 KV | 131 072 | **33.9 tok/s** @128K, 58.7 @8K | нет MTP, нет нативного vision |
| Форк aljaz (A5000) | RTX A5000 Laptop 16 ГБ | NInfer sm_86, min-Q4 | 122 880 | **22.4 tok/s** (MTP3), 12.6 plain @131K | первый 16-ГБ порт |
| **Данный форк** | **RTX 5080 16 ГБ** | **NInfer sm_120a, min-Q4 + MTP-Q4, `i4` KV, chunk 256** | **124 928** | **71–144 tok/s** (MTP3), 53 MTP0 | Vision, prefill chunk 256 |

Сравнение на одном железе (RTX 5080, llama.cpp против данного форка, длинная генерация):

| Контекст | llama.cpp decode | данный форк decode (MTP3) | prefill данного форка |
|---:|---:|---:|---:|
| ~32K | ~51.5 tok/s | 117–133 tok/s (код/перевод), 71–86 (проза) | 1 500 tok/s |
| ~96K | ~40 tok/s | ~80–112 tok/s | 1 116 tok/s |
| ~123K | 33.9 tok/s | ~86 tok/s (проза), до 129 (короткие ответы) | 1 017 tok/s |

Decode сильно зависит от контента (acceptance 31–99 %). Structured output даёт
~144 tok/s, проза/истории — ~71–86 tok/s. `gsm8k_cot` (8-shot, lm-evaluation-harness,
50 семплов): **0.96**; RULER NIAH retrieval на 8K/32K/64K: 3/3.

## Установка

Сборка образа рантайма (CUDA 13.1, `sm_120a`):

```bash
docker build -t ninfer-aljaz:build -f Dockerfile.aljaz .
docker build -t ninfer-mtpq4:build -f Dockerfile.mtpq4 .
```

Далее по [docs/BUILDING.ru.md](docs/BUILDING.ru.md) — конвертация
`Qwen/Qwen3.8-27B` в артефакт min-Q4/MTP-Q4/Vision-Q4, либо скачивание готового
артефакта с Hugging Face.

## Документация

| Документ | EN | RU |
|---|---|---|
| Обзор | [README.md](README.md) | [README.ru.md](README.ru.md) |
| Полные результаты, методика, сырые данные | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | [docs/BENCHMARKS.ru.md](docs/BENCHMARKS.ru.md) |
| Рецепт сборки/квантования и обоснование решений | [docs/BUILDING.md](docs/BUILDING.md) | [docs/BUILDING.ru.md](docs/BUILDING.ru.md) |

## Статус и ограничения

- Проверено на RTX 5080 16 ГБ. Другие 16-ГБ `sm_120`-карты — **ожидаемо
  совместимы, на железе не проверялись**.
- Безопасный потолок text + MTP3 — 124 928 токенов (OOM между 128 000 и 129 024);
  MTP0 доходит до 131 072. С `--prefill-chunk 256` потолок 124 928
  (абсолютный — 125 952).
- Vision ограничен 32 768 merged-токенами рантайма и несовместим с текстовым
  профилем на 124 928.
- Greedy-вывод MTP3 побитово совпадает с MTP0 на коротких ответах и может
  расходиться на длинных генерациях в near-tie (оба варианта когерентны).
- `--prefill-chunk 256` — производительный профиль: он может менять точный
  выбор токенов против chunk 64 в near-tie (скорость decode не меняется).
- `--lm-head-draft` не помещается на потолке 124 928, нужен меньший контекст.

## Лицензия и атрибуция

- Базовая модель: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0.
- Рантайм: [Neroued/ninfer](https://github.com/Neroued/ninfer).
- 16-ГБ min-Q4 и порт sm_86: [aljazceru/ninfer](https://github.com/aljazceru/ninfer).
- Blackwell MTP-Q4/Vision-Q4, 16-ГБ профиль и проверка: данный форк.

Apache-2.0. Собственная лицензия и документация апстрима сохранены.
