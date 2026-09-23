import os
import sys
import time
from pathlib import Path
import yt_dlp
from dotenv import load_dotenv

# Опционально, но лучше создать акк на 'huggingface.co' и добавить ключ-токен
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from faster_whisper import WhisperModel



# ─────────────────────────────────────────────────────────────
#  Настройки — меняйте тут
# ─────────────────────────────────────────────────────────────
VIDEO_URL = "https://www.youtube.com/watch?v=TtqItIFoaLI&t=2s"   # ← ваша ссылка
MODEL_SIZE   = "base"      # tiny / base / small / medium / large-v3 / turbo
COMPUTE_TYPE = "int8"       # int8 (CPU) / float16 (GPU) / int8_float16
DEVICE       = "cpu"        # cpu / cuda
LANGUAGE     = "ru"         # ru / en / None (автоопределение)
OUTPUT_FILE  = "transcript.txt"
# ─────────────────────────────────────────────────────────────


def human_time(seconds: float) -> str:
    """Форматирует секунды как ЧЧ:ММ:СС"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def download_audio(video_url: str) -> Path:
    """Скачивает видео и возвращает путь к извлечённому mp3"""
    print(f"📥 Скачивание аудио: {video_url}")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "temp_audio.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        audio_file = Path(os.path.splitext(filename)[0] + ".mp3")

    if not audio_file.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_file}")

    size_mb = audio_file.stat().st_size / 1024 / 1024
    print(f"✅ Аудио сохранено: {audio_file.name} ({size_mb:.1f} МБ)\n")
    return audio_file


def draw_progress(audio_pos: float, total: float, elapsed: float,
                  segments_count: int) -> None:
    """Рисует строку прогресса в одной строке терминала."""
    # Защита от деления на ноль
    total = max(total, 0.001)
    percent = min(audio_pos / total, 1.0)
    bar_len = 30
    filled = int(bar_len * percent)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Оценка оставшегося времени
    if percent > 0.01:
        eta = elapsed * (1 - percent) / percent
        eta_str = human_time(eta)
    else:
        eta_str = "--:--"

    line = (
        f"\r[{bar}] {percent*100:5.1f}%  "
        f"аудио {human_time(audio_pos)}/{human_time(total)}  "
        f"прошло {human_time(elapsed)}  осталось ~{eta_str}  "
        f"сегментов {segments_count}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def transcribe(audio_file: Path, model_size: str) -> str:
    """Загружает модель и транскрибирует аудио с прогресс-баром."""

    print(f"🧠 Загрузка модели '{model_size}' ({DEVICE}, {COMPUTE_TYPE})...")
    load_start = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"✅ Модель загружена за {time.time() - load_start:.1f} сек\n")

    print("🎧 Транскрибация началась. Не закрывайте окно...\n")

    transcribe_start = time.time()

    # transcribe возвращает генератор сегментов + объект с метаданными.
    # Аудио декодируется лениво, поэтому мы можем показывать прогресс.
    segments, info = model.transcribe(
        str(audio_file),
        language=LANGUAGE,
        beam_size=5,  # компромисс между качеством и скоростью
        vad_filter=True,  # отсекает паузы и тишину
        vad_parameters={"min_silence_duration_ms": 500},
    )

    total_duration = info.duration
    print(f"📊 Длительность аудио: {human_time(total_duration)}")

    full_text_parts: list[str] = []
    segments_count = 0
    last_segment_end = 0.0

    # Потоково читаем сегменты и обновляем прогресс
    for segment in segments:
        text = segment.text.strip()
        if text:
            full_text_parts.append(text)
            segments_count += 1
        last_segment_end = segment.end

        # Обновляем прогресс-бар (без переноса строки)
        elapsed = time.time() - transcribe_start
        draw_progress(last_segment_end, total_duration, elapsed, segments_count)

    print()  # перевод строки после прогресс-бара

    total_elapsed = time.time() - transcribe_start
    speed_ratio = total_duration / total_elapsed if total_elapsed > 0 else 0

    print(f"\n✅ Транскрибация завершена за {human_time(total_elapsed)}")
    print(f"⚡ Соотношение скорость/длительность -- 1:{speed_ratio:.1f}")
    return " ".join(full_text_parts)


def main():
    # Проверка FFmpeg (нужен только для yt-dlp)
    if os.system("ffmpeg -version >nul 2>&1" if os.name == "nt"
                 else "ffmpeg -version >/dev/null 2>&1") != 0:
        print("❌ FFmpeg не найден в PATH. Установите его и перезапустите терминал.")
        sys.exit(1)

    audio_file = None
    try:
        audio_file = download_audio(VIDEO_URL)
        text = transcribe(audio_file, MODEL_SIZE)

        # Сохраняем результат
        out_path = Path(OUTPUT_FILE)
        out_path.write_text(text, encoding="utf-8")

        print(f"\n💾 Текст сохранён в: {out_path.resolve()}")
        print(f"📝 Символов: {len(text):,}  |  Слов (примерно): {len(text.split()):,}")

    except KeyboardInterrupt:
        print("\n\n⛔ Прервано пользователем.")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        sys.exit(1)
    finally:
        # Убираем временный mp3
        if audio_file and audio_file.exists():
            try:
                audio_file.unlink()
                print("🧹 Временный аудиофайл удалён.")
            except OSError:
                pass


if __name__ == "__main__":
    main()
