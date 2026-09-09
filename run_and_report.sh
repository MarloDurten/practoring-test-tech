#!/usr/bin/env bash
# Обработать видео, сохранить маркировки и только ПОСЛЕ этого вывести отчёт в терминал.
#
# Примеры:
#   ./run_and_report.sh exam.mp4
#   ./run_and_report.sh exam.mp4 --profile fast
#   ./run_and_report.sh exam.mp4 --profile balanced --max-duration 60
#   ./run_and_report.sh 0 --profile fast --max-duration 20
#
# Работает в Git Bash / WSL / Linux / macOS.

set -euo pipefail

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-C.UTF-8}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Использование:
  ./run_and_report.sh <видео|индекс_камеры> [опции приложения...]

Опции скрипта:
  -h, --help          эта справка
  --output-dir DIR    каталог отчётов (по умолчанию reports)
  --no-mark           не писать marked-видео, только CSV/JSON
  --keep-logs         не глушить логи анализатора

Все остальные аргументы передаются в python -m app.main
(например --profile fast, --max-duration 30, --no-gaze).
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SOURCE="$1"
shift

OUTPUT_DIR="reports"
SAVE_MARKED=1
QUIET_APP=1
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --no-mark)
      SAVE_MARKED=0
      shift
      ;;
    --keep-logs)
      QUIET_APP=0
      shift
      ;;
    *)
      PASSTHRU+=("$1")
      shift
      ;;
  esac
done

if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

mkdir -p "$OUTPUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$OUTPUT_DIR/run_${STAMP}.log"

CMD=("$PYTHON" -m app.main --source "$SOURCE" --output-dir "$OUTPUT_DIR")
if [[ "$SAVE_MARKED" -eq 1 ]]; then
  CMD+=(--save-marked)
fi
if [[ "$QUIET_APP" -eq 1 ]]; then
  CMD+=(--log-level WARNING)
fi
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  CMD+=("${PASSTHRU[@]}")
fi

echo "============================================================"
echo "  Обработка видео и создание маркировок"
echo "============================================================"
echo "Источник:   $SOURCE"
echo "Python:     $PYTHON"
echo "Отчёты:     $OUTPUT_DIR"
echo "Команда:    ${CMD[*]}"
echo "Лог прогона:$RUN_LOG"
echo "------------------------------------------------------------"
echo "Идёт анализ... (отчёт появится после завершения)"
echo

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [[ $STATUS -ne 0 ]]; then
  echo
  echo "ОШИБКА: анализ завершился с кодом $STATUS. Смотрите $RUN_LOG" >&2
  exit "$STATUS"
fi

# Берём папку конкретного прогона (не смешиваем отчёты разных роликов).
REPORT_DIR="$(
  grep -E '^REPORT_DIR:' "$RUN_LOG" | tail -n 1 | sed 's/^REPORT_DIR:[[:space:]]*//' | tr -d '\r'
)"

if [[ -z "$REPORT_DIR" ]]; then
  JSON="$("$PYTHON" -c "from pathlib import Path; p=sorted(Path(r'$OUTPUT_DIR').rglob('summary_*.json'), key=lambda x: x.stat().st_mtime, reverse=True); print(p[0] if p else '')")"
  CSV="$("$PYTHON" -c "from pathlib import Path; p=sorted(Path(r'$OUTPUT_DIR').rglob('frames_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True); print(p[0] if p else '')")"
  MARKED="$("$PYTHON" -c "from pathlib import Path; p=sorted(Path(r'$OUTPUT_DIR').rglob('marked_*.mp4'), key=lambda x: x.stat().st_mtime, reverse=True); print(p[0] if p else '')")"
else
  if [[ -d "$REPORT_DIR/logs" ]]; then
    cp -f "$RUN_LOG" "$REPORT_DIR/logs/launcher.log" 2>/dev/null || true
  fi
  JSON="$(ls -1t "$REPORT_DIR"/json/summary_*.json 2>/dev/null | head -n 1 || true)"
  CSV="$(ls -1t "$REPORT_DIR"/csv/frames_*.csv 2>/dev/null | head -n 1 || true)"
  MARKED="$(ls -1t "$REPORT_DIR"/marked/marked_*.mp4 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$JSON" || ! -f "$JSON" ]]; then
  echo "Не найден JSON-отчёт в $OUTPUT_DIR после прогона." >&2
  exit 3
fi

echo
echo "============================================================"
echo "  РЕЗУЛЬТАТЫ ПОСЛЕ ОБРАБОТКИ И МАРКИРОВОК"
echo "============================================================"
echo

"$PYTHON" - "$JSON" "$CSV" "$MARKED" <<'PY'
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

json_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
marked_path = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

data = json.loads(json_path.read_text(encoding="utf-8"))
lat = data.get("latency_ms") or {}
cpu = data.get("cpu_process_percent") or {}
ram = data.get("ram_mb") or {}
events = data.get("events") or {}
gpu = data.get("gpu")

def fmt(value):
    return "—" if value in (None, "", []) else value

print(f"Источник:             {fmt(data.get('source'))}")
print(f"Профиль:              {fmt(data.get('profile'))}")
print(f"Source resolution:    {fmt(data.get('source_resolution'))}")
print(f"Working resolution:   {fmt(data.get('working_resolution'))}")
print(f"Кадров прочитано:     {fmt(data.get('frames_read'))}")
print(f"Кадров обработано:    {fmt(data.get('frames_processed'))}")
print(f"Кадров пропущено:     {fmt(data.get('frames_skipped'))}")
print(f"Суммарное время, с:   {fmt(data.get('wall_time_sec'))}")
print(
    "Latency mean/med/p95: "
    f"{fmt(lat.get('mean'))} / {fmt(lat.get('median'))} / {fmt(lat.get('p95'))} мс"
)
print(f"Средний FPS (proc):   {fmt(data.get('avg_fps_processed'))}")
print(f"CPU process avg/max:  {fmt(cpu.get('avg'))} / {fmt(cpu.get('max'))} %")
print(f"RAM avg/peak, МБ:     {fmt(ram.get('avg'))} / {fmt(ram.get('peak'))}")
if gpu:
    print(
        "GPU util avg/max:     "
        f"{fmt(gpu.get('util_percent_avg'))} / {fmt(gpu.get('util_percent_max'))} %"
    )
else:
    print("GPU:                  CPU-only / не измерялся")
print(
    "Лицо найдено:         "
    f"{fmt(events.get('face_detected_frames'))} ({fmt(events.get('face_detected_ratio'))})"
)
print(
    "Отвлечение:           "
    f"{fmt(events.get('distraction_frames'))} ({fmt(events.get('distraction_ratio'))})"
)
print(
    "AI-подозрительные:    "
    f"{fmt(events.get('ai_suspicious_frames'))} ({fmt(events.get('ai_suspicious_ratio'))})"
)

print()
print("Файлы:")
print(f"  JSON:   {json_path}")
print(f"  CSV:    {csv_path if csv_path and csv_path.exists() else '—'}")
print(f"  MARKED: {marked_path if marked_path and marked_path.exists() else 'не создан'}")

if csv_path and csv_path.exists():
    distraction_rows = []
    ai_rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("skipped") == "1":
                continue
            idx = row.get("source_frame_index", "")
            ts = row.get("timestamp_sec", "")
            if row.get("distraction") == "1":
                distraction_rows.append((idx, ts, row.get("distraction_reasons", "")))
            if row.get("ai_suspicious") == "1":
                ai_rows.append((idx, ts, row.get("ai_signals", ""), row.get("ai_suspicion_score", "")))

    print()
    print(f"Маркировки отвлечения ({len(distraction_rows)} кадров):")
    if not distraction_rows:
        print("  нет")
    else:
        for idx, ts, reasons in distraction_rows[:25]:
            print(f"  frame={idx}  t={ts}s  {reasons}")
        if len(distraction_rows) > 25:
            print(f"  ... ещё {len(distraction_rows) - 25} (см. CSV)")

    print()
    print(f"Маркировки AI-подозрительных кадров ({len(ai_rows)}):")
    if not ai_rows:
        print("  нет")
    else:
        for idx, ts, signals, score in ai_rows[:25]:
            print(f"  frame={idx}  t={ts}s  score={score}  {signals}")
        if len(ai_rows) > 25:
            print(f"  ... ещё {len(ai_rows) - 25} (см. CSV)")
PY

echo
echo "Готово."
