# Сборка и квантование 16-гигабайтного артефакта Qwen3.8-27B

Подробный рецепт преобразования `Qwen/Qwen3.8-27B` в NInfer-артефакт этого
форка, а также обоснование каждого решения. Готовый артефакт опубликован:
[`mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080`](https://huggingface.co/mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080)
(`qwen3_8_27b_minq4_mtpq4_visionq4.ninfer`, 15 514 935 296 байт, sha256
`c7eb6fdde74bfd70e168a9ccce113b08fb34ad7ce3349ea989590c4830591fb8`).

## 0. Ограничения, задающие дизайн

Модель 27B должна поместиться вместе с рабочим KV-кэшем, спекулятивной и
vision-головками в **16 303 MiB** памяти GPU:

| Компонент | Бюджет |
|---|---|
| Веса текста | насколько позволяет качество |
| MTP-голова | обязана поместиться рядом с текстом |
| Vision | нужен, но только при `--vision` |
| KV-пул | максимально большой — контекст и есть продукт |

Поэтому форк сводит все крупные тензоры к Q4 (`Q4G64_F16S`), а Vision-буферы
делает ленивыми, оставляя KV-пул главным потребителем VRAM.

## 1. Требования к хосту

- Linux x86_64, NVIDIA RTX 50-серии (`sm_120`), драйвер >= 610.57
- Docker + NVIDIA Container Toolkit
- ~60 ГБ свободного диска (bf16 ~56 ГБ + базовый + финальный артефакт)
- Python 3.10+ с `torch`, `safetensors`, `numpy`, `transformers` для конвертации

## 2. Сборка образа рантайма

Из корня репозитория (CUDA 13.1, `sm_120a`):

```bash
docker build -t ninfer-aljaz:build -f Dockerfile.aljaz .
docker build -t ninfer-mtpq4:build -f Dockerfile.mtpq4 .
```

`Dockerfile.aljaz` берёт `nvidia/cuda:13.1.2-devel-ubuntu24.04` и настраивает
CMake с `-DCMAKE_CUDA_ARCHITECTURES=120a`. `Dockerfile.mtpq4` пересобирает
`ninfer` и `ninfer-serve` поверх него. Образ кампании — `ninfer-mtpq4:build`
`sha256:723834522026fb327a54b04aed7b6c9a6421b19e4596677e85efba7a9981445a`.

## 3. Загрузка исходной модели

```bash
huggingface-cli download Qwen/Qwen3.8-27B --local-dir Qwen3.8-27B
```

Конвертер проверяет официальные SHA-256 шести frontend-ресурсов
(`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`,
`video_preprocessor_config.json`).

## 4. Конвертация в базовый min-Q4 артефакт

`tools/convert/qwen3_8_27b/inventory.py` форка переопределяет раскладку на
минимальную all-Q4:

- MLP, входные и выходные проекции GDN/attention -> Q4;
- слитые Q4/Q4 входные ядра для GDN и attention;
- Q4 `linear_add` для residual;
- Q4 выходная голова;
- эмбеддинг токенов остаётся Q6.

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model Qwen3.8-27B \
  --out out/qwen3_8_27b_minq4.ninfer \
  --device cuda
```

Результат: `out/qwen3_8_27b_minq4.ninfer` (~15.1 ГБ). MTP и Vision пока W8/Q5.

**Почему такая раскладка.** Контроль качества — эмулятор PPL
(`tools/eval/sim_ppl.py`): он применяет ровно ту же groupwise-математику
квантования и меряет LM loss на Wikitext-2 (50x512) относительно якоря
IQ3_XXS (PPL 6.2569). Раскладка all-Q4 прошла ворота на **+0.04 PPL**;
более агрессивный Q3-MLP провалился (+0.79) и отвергнут. Эмбеддинг оставлен
Q6, потому что Q4-gather-пути нет, а тензор чувствителен к качеству.

## 5. Переквантование MTP и Vision в Q4 (выравнивание 256 байт)

Каждый тензор выравнивается по 256 байт, payload — по 4096 байт; иначе рантайм
падает с `tensor is not 256-byte aligned`.

```bash
# 5a. Матрицы MTP -> Q4G64_F16S
python3 tools/artifact/repack_mtp_q4.py \
  --src out/qwen3_8_27b_minq4.ninfer \
  --dst out/qwen3_8_27b_minq4_mtpq4.ninfer

# 5b. Матрицы Vision tower -> Q4G64_F16S
python3 tools/artifact/repack_vision_q4.py \
  --src out/qwen3_8_27b_minq4_mtpq4.ninfer \
  --dst out/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
```

**Зачем.** MTP и Vision в зарегистрированной раскладке хранятся в W8/Q5; их
перевод в Q4 освобождает несколько сотен МиБ, которые уходят прямо в KV-пул —
именно это даёт контекст 124 928. Репакеры трогают только эти тензоры, тело
текста переносится байт-в-байт.

**Зачем ленивый Vision.** При жадном выделении Vision-workspace съедает VRAM
даже в текстовых прогонах и снижает потолок контекста. Форк выделяет и
освобождает его по требованию, поэтому текст сохраняет полный пул, а Vision
платит только при использовании.

## 6. Проверка

```bash
python3 tools/artifact/verify_artifacts.py out/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
bash tools/artifact/run_load_verify.sh out/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
```

Опубликованный артефакт дополнительно проверен сквозным прогоном: грузится за
~12 с, сообщает `target=qwen3_8_27b`, `weights_id=groupwise-int`, 1 115 тензоров
и 7 ресурсов, и прошёл всю сервинг-кампанию в
[`../validation/5080-20260918`](../validation/5080-20260918).

## 7. Запуск

```bash
docker run --rm --gpus all --network host \
  -v "$PWD/models:/ninfer-models:ro" \
  ninfer-mtpq4:build \
  ninfer-serve /ninfer-models/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --model-id qwen3.8-27b-16gb \
  --host 0.0.0.0 --port 18100 \
  --max-context 124928 --kv-capacity 124928 \
  --kv-dtype i4 --spec mtp --draft-tokens 3 \
  --prefill-chunk 256 --no-cuda-graph
```

## 8. Почему такие дефолты

Каждый параметр измерен на RTX 5080 16 ГБ; см. [BENCHMARKS.ru.md](BENCHMARKS.ru.md).

| Параметр | Выбор | Причина |
|---|---|---|
| `--prefill-chunk` | **256** (было 64) | prefill ×2.35–2.53, +34 МиБ; 512 — OOM |
| `--spec mtp --draft-tokens` | **3** | оптимум throughput; MTP2 медленнее; >5 не поддерживается |
| `--kv-dtype` | **i4** (int4-group128) | быстрее всех и выше acceptance; `int8`/`bf16` — OOM на 64K |
| `--max-context`/`--kv-capacity` | **124928** | безопасный потолок с запасом; MTP3 OOM между 128 000 и 129 024 |
| `--no-cuda-graph` | да | CUDA Graphs не дали измеримого выигрыша |
| `--no-prefix-reuse` | только бенчмарки | каждый промпт считается целиком, честные цифры |
| `--vision` | отдельный режим | нужен меньший пул; рантайм ограничивает Vision 32 768 токенами |

## 9. Происхождение

Опубликованный артефакт собран из ревизии форка
`024b3ea4b91b67fdd75d8ca947e2a58a4258237b` плюс патч рабочего дерева
(`validation/5080-20260918/fork-working-tree.patch`, sha256 диффа
`d7259b5ac59b8a35fef685c549ec268944e62472bb0f63dd8c6f46434e3d1a13`) на образе
`ninfer-mtpq4:build` `sha256:723834522026fb327a54b04aed7b6c9a6421b19e4596677e85efba7a9981445a`.

Apache-2.0. Базовая модель: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Рантайм: [Neroued/ninfer](https://github.com/Neroued/ninfer). 16-ГБ min-Q4 и
порт sm_86: [aljazceru/ninfer](https://github.com/aljazceru/ninfer).
