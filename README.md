# Proctoring Load Test

Тестовое приложение для **оценки серверной нагрузки** при анализе видео с веб-камеры или файла. Оно измеряет CPU/RAM/latency и параллельно считает два типа сигналов:

1. **Отвлечение от экрана** — лицо пропало, взгляд/голова уведены, аномальные моргания.
2. **Подозрительные кадры** — дешёвые эвристики, которыми можно отбирать кадры на последующую проверку AI-generated content. Это **не** финальный детектор дипфейка.

Разрешение камеры или ролика вручную подбирать не нужно. Любой входной кадр автоматически приводится к внутреннему стандарту выбранного профиля.

## Возможности

- Захват с веб-камеры или видеофайла (до 10 минут).
- Нормализация: letterbox или center-crop, сохранение aspect ratio.
- Профили `high_quality` / `balanced` / `fast`.
- Модульный пайплайн: захват → нормализация → предобработка → CV-инференс → постобработка → метрики.
- Бенчмарк: время кадра, mean/median/p95 latency, FPS, CPU, RAM, опционально GPU/VRAM.
- Отчёты: CSV по кадрам, JSON-итог, консольный summary.
- Переключатели модулей: face tracking, gaze, distraction, AI-эвристики.
- CPU-only, если GPU нет.

## Структура

```
app/
  main.py                 # CLI
  video_capture.py        # камера / файл, без доверия к cap.set()
  normalization.py        # внутреннее разрешение + letterbox/crop
  metrics.py              # latency, CPU, RAM, GPU
  analyzers/
    preprocessing.py
    face_tracking.py      # MediaPipe Face Mesh
    gaze_estimation.py
    distraction.py
    ai_heuristics.py
    postprocessing.py
    pipeline.py
tests/
requirements.txt
README.md
```

## Профили качества

| Профиль         | Рабочее разрешение | Кадры        | Назначение                          |
|-----------------|--------------------|--------------|-------------------------------------|
| `high_quality`  | 1280×720           | каждый       | верхняя оценка нагрузки             |
| `balanced`      | 640×480            | 1 из 3       | типичный серверный режим            |
| `fast`          | 640×360            | 1 из 5       | нижняя оценка / запас по CPU        |

Исходное разрешение и рабочее логируются **отдельно**. Если вход больше ~1080p (например 4K), перед инференсом выполняется дополнительный downscale. `cap.set(WIDTH/HEIGHT)` используется только как подсказка драйверу и никогда как источник истины.

## Зависимости и альтернативы

Рекомендуемый runtime: **Python 3.11 или 3.12**.

MediaPipe 0.10.30+ перешёл на **Tasks API** (`FaceLandmarker`) и больше не содержит `mp.solutions.face_mesh`. Приложение поддерживает оба варианта: старый `solutions`, если он есть, и Tasks с автоскачиванием `face_landmarker.task` в `app/models/` при первом запуске (нужен интернет).

### Обязательные

| Библиотека       | Зачем                                      | Тяжёлая установка? |
|------------------|--------------------------------------------|--------------------|
| `opencv-contrib-python` | захват, resize, EAR/pose, артефакты. Не ставьте рядом `opencv-python` — два пакета `cv2` конфликтуют. | нет |
| `mediapipe`      | FaceLandmarker / Face Mesh, landmarks, iris. Первый запуск скачивает ~4 МБ `.task` | умеренно |
| `numpy`          | массивы кадров                             | нет                |
| `psutil`         | CPU / RAM процесса                          | нет                |

### Опциональные

| Библиотека     | Зачем                         | Когда ставить                          |
|----------------|-------------------------------|----------------------------------------|
| `nvidia-ml-py` | GPU util + VRAM              | есть NVIDIA GPU; иначе можно не ставить |
| `pytest`       | юнит-тесты                    | разработка                             |

### Что сознательно не стоит в `requirements.txt`

- **Py-Feat** — даёт AU, эмоции и калиброванный gaze, но тянет PyTorch, sklearn, большие веса моделей и часто ломается на Windows. Для нагрузочного теста достаточно iris + head pose из MediaPipe.
- **LibreFace** — похожая история: отдельный чекпойнт и GPU-стек. Имеет смысл только если вы уже меряете именно модель эмоций, а не инфраструктуру захвата/нормализации.

Если позже понадобятся AU/эмоции, подключайте их как отдельный модуль с тем же `enabled`-флагом, не смешивая с бенчмарком захвата.

## Установка

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

При первом запуске с включённым face tracking приложение скачает `face_landmarker.task` (~4 МБ) в `app/models/`. Без интернета положите файл туда вручную (ссылка в `app/models/README.md`).

Проверка:

```bash
python -m pytest
```

## Интерактивный запуск в PowerShell

В папке с видео (или в папке проекта) стрелками выбираете файл и формат сжатия. После проверки в том же окне печатается отчёт: длительность ролика, сколько шла проверка, CPU/RAM/GPU.

```powershell
# из папки с видео
cd C:\путь\к\видео
C:\Users\user\Desktop\practoring_test_tech\start.ps1

# или из папки проекта
cd C:\Users\user\Desktop\practoring_test_tech
.\start.ps1
```

Управление: **↑ / ↓** выбрать, **Enter** подтвердить, **Q** выход.

Порядок меню:
1. видео подходящего формата (`mp4`, `mkv`, `avi`, `mov`, `webm`, …);
2. профиль качества (`balanced` / `high_quality` / `fast`);
3. сжатие marked-ролика (MP4 MPEG-4, H.264, AVI XVID, MJPEG).

То же без `start.ps1`:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m app
```

Если нужен старый CLI без меню:

```powershell
.\.venv\Scripts\python.exe -m app.main --source ".\2026-09-09 14-42-26.mkv" --profile balanced --save-marked --codec mp4v
```

## Запуск через bash-скрипт (отчёт после маркировок)

`run_and_report.sh` обрабатывает видео, сохраняет marked-ролик и **только после этого** печатает сводку в терминал: нагрузка, отвлечения, AI-сигналы и пути к файлам.

```bash
# Git Bash / WSL / Linux / macOS, из корня репозитория
chmod +x run_and_report.sh
./run_and_report.sh exam.mp4
./run_and_report.sh exam.mp4 --profile fast --max-duration 30
./run_and_report.sh 0 --profile balanced --max-duration 20
```

На Windows в PowerShell, если установлен Git Bash:

```powershell
bash .\run_and_report.sh sample_clip.mp4 --profile fast --max-frames 30
```

Опции скрипта: `--output-dir DIR`, `--no-mark` (без marked-видео), `--keep-logs`. Остальные флаги уходят в `python -m app.main`.

Marked-видео можно сохранить и напрямую:

```bash
python -m app.main --source exam.mp4 --profile balanced --save-marked
```

## Запуск на видеофайле

```bash
python -m app.main --source path\to\video.mp4 --profile balanced
```

Примеры:

```bash
# Максимальная нагрузка, все кадры, 1280x720
python -m app.main --source exam.mp4 --profile high_quality --output-dir reports

# Быстрый прогон 30 секунд
python -m app.main --source exam.mp4 --profile fast --max-duration 30

# Center-crop вместо letterbox
python -m app.main --source exam.mp4 --profile balanced --resize-mode center_crop
```

Поддерживаются обычные контейнеры, которые открывает OpenCV (`mp4`, `avi`, `mkv`…). Длительность больше 10 минут обрезается.

## Запуск с веб-камеры

```bash
python -m app.main --source 0 --profile balanced
```

Вторая камера: `--source 1`. Локальный preview:

```bash
python -m app.main --source 0 --profile fast --preview
```

Выход из preview: `q` или `Esc`.

Разрешение камеры в драйвере менять не нужно. Приложение читает фактический кадр и само приводит его к профилю.

## Переключение модулей

```bash
python -m app.main --source exam.mp4 --no-gaze
python -m app.main --source exam.mp4 --no-ai-heuristics
python -m app.main --source exam.mp4 --no-face --no-gaze
```

Флаги:

- `--no-face` — выключить face tracking (MediaPipe).
- `--no-gaze` — выключить оценку взгляда.
- `--no-distraction` — выключить эвристики отвлечения.
- `--no-ai-heuristics` — выключить сигналы AI-generated кадров.

Так можно отдельно измерить вклад каждой стадии в CPU.

## Отчёты

После прогона файлы **не сваливаются в одну кучу**. Корень `reports/` делится по источнику и по запуску:

```
reports/
  2026-09-09_14-42-26/          # имя видео (пробелы → _)
    20260909_145000/            # конкретный прогон
      csv/frames_balanced.csv
      json/summary_balanced.json
      marked/marked_balanced.mp4
      logs/
  camera_0/
    20260909_144533/
      csv/
      json/
      marked/
```

Повторный анализ того же файла создаёт **новую** папку со штампом времени. Старый `--flat-output` пишет всё прямо в `--output-dir`.

### Как читать CSV

Ключевые колонки:

- `source_width/height` vs `working_width/height` — вход и внутренний стандарт.
- `process_ms` — полное время обработки кадра.
- `infer_ms` — только CV-инференс.
- `skipped=1` — кадр прочитан, но не анализировался (stride профиля).
- `cpu_process_percent`, `ram_mb`, опционально `gpu_util_percent` / `vram_mb`.
- `distraction`, `distraction_reasons` — сработали правила отвлечения.
- `ai_suspicion_score`, `ai_suspicious`, `ai_signals` — pre-filter, не вердикт.

### Как интерпретировать результаты

- **`avg_fps_processed`** — сколько проанализированных кадров в секунду реально вытягивает машина. Если он заметно ниже целевого FPS источника / stride, сервер не успевает.
- **`latency_ms.p95`** — хвост задержки. Для онлайн-прокторинга ориентир: p95 инференса должен укладываться в бюджет кадра с запасом (например < 30–50 мс на `balanced` CPU, если цель ~10 анализируемых FPS).
- **`cpu_process_percent`** — загрузка процесса. На многоядерной машине 100% ≈ одно ядро. Для сервера смотрите, сколько таких воркеров влезает: `cores * 0.7 / (avg_cpu/100)`.
- **`ram_mb.peak`** — пик RSS. Умножьте на число параллельных сессий и добавьте ОС/очередь кадров.
- **`face_detected_ratio`** — если на нормальном ролике с человеком близко к 0, проблема в камере/освещении, а не в нагрузке.
- **`distraction_*`** — эвристики, не ground truth. Используйте как нагрузку на ветку правил и как sanity-check пайплайна.
- **`ai_suspicious_ratio`** — доля кадров, которые имеет смысл отправить в более тяжёлую forensic-модель. Сам score не равен «это дипфейк».

### Как оценивать серверную нагрузку

Практичный протокол:

1. Прогоните один и тот же ролик (~2–10 мин) на трёх профилях.
2. Зафиксируйте `wall_time_sec`, `latency_ms.p95`, `avg_fps_processed`, `cpu_process_percent.avg`, `ram_mb.peak`.
3. Оцените ёмкость:  
   `sessions ≈ (CPU_cores * target_cpu_headroom) / (cpu_process_percent.avg / 100)`  
   и проверьте, что `avg_fps_processed` не ниже нужного (для `balanced` это примерно FPS_источника / 3).
4. Повторите с `--no-ai-heuristics` и `--no-gaze`, чтобы увидеть вклад модулей.
5. Если есть GPU — сравните CPU-only vs GPU. MediaPipe Face Mesh в этом приложении идёт на CPU; GPU-метрики нужны как база, когда вы добавите нейросеть детекции AI-контента.

Пример: 8 vCPU, `balanced`, CPU avg 85% на одну сессию, p95 = 40 мс, processed FPS = 10 при источнике 30 FPS. Запас по FPS есть, по CPU — нет: параллельно потянете примерно 8 * 0.7 / 0.85 ≈ 6 сессий, если не упрётесь в RAM.

## Эвристики отвлечения

Срабатывают, если:

- лицо отсутствует N обработанных кадров подряд;
- yaw/pitch взгляда больше порога;
- голова повёрнута / наклонена больше порога;
- глаза закрыты слишком долго или частота морганий аномальна (скользящее окно).

Пороги задаются в `app/config.py` (`DistractionThresholds`).

## Сигналы AI-generated кадра

Не классификатор, а набор индикаторов:

- нет стабильных landmarks;
- низкая уверенность трекинга;
- резкий скачок координат между кадрами;
- нестабильность landmarks в окне;
- лицо слишком «пластиковое» (низкая Laplacian variance) или слишком шумное относительно фона.

Подозрительные кадры можно складывать в очередь на отдельный детектор (CNN / forensic).

## Расширение под production

Краткая карта, без переписывания ядра:

1. **Вынести инференс из CLI в сервис** — gRPC/HTTP воркер, кадры из Redis/NATS, один процесс = одна сессия или батч.
2. **Очередь и backpressure** — не анализировать быстрее, чем успевает воркер; при перегрузе повышать stride (adaptive `fast`).
3. **Заменить эвристики взгляда/эмоций** на калиброванную модель (Py-Feat / собственный gaze CNN) за тем же интерфейсом `enabled` + `infer()`.
4. **AI-контент** — оставить текущие сигналы как pre-filter, на `ai_suspicious=True` слать кроп лица в отдельный GPU-сервис.
5. **Идентичность и аудит** — не хранить сырое видео дольше политики; в логах только метрики и коды событий.
6. **Наблюдаемость** — прокинуть `process_ms` / CPU / очередь в Prometheus; CSV оставить для офлайн-бенчей.
7. **Модели MediaPipe** — прогрев при старте пода, healthcheck после warmup.
8. **Мультифейс и антиспуф** — Face Detection + anti-spoof до Mesh, чтобы не кормить Mesh скриншотами.
9. **Конфиг** — пороги и профили из YAML/ENV, не из кода.
10. **Нагрузочный контур** — этот CLI гонять в CI на фиксированном ролике и падать, если p95/CPU выросли относительно baseline.

## Лицензии сторонних библиотек

OpenCV, MediaPipe, NumPy, psutil — свои open-source лицензии. Перед коммерческим прокторингом проверьте условия MediaPipe/TensorFlow и политику обработки биометрии в вашей юрисдикции.
