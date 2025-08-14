import os
import tempfile
import math
import subprocess
import streamlit as st
import whisper

# ======= Настройки =======
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # tiny, base, small, medium, large
MODEL_DIR = os.path.join(os.getcwd(), "models")    # папка для моделей
os.makedirs(MODEL_DIR, exist_ok=True)
CHUNK_LENGTH_SEC = 300  # длина куска при нарезке больших файлов (в секундах)
# =========================

# Загружаем модель Whisper из локальной папки (кэшируем)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model(MODEL_NAME, download_root=MODEL_DIR)

model = load_whisper_model()

def convert_to_wav(input_path):
    """Конвертирует любой аудиофайл в WAV 16kHz mono с помощью ffmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        tmp_wav.name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_wav.name

def transcribe_with_whisper(file_path):
    """Распознаёт весь файл с помощью Whisper."""
    result = model.transcribe(file_path, language="ru")
    return result["text"]

def get_audio_duration(path):
    """Получает длительность аудиофайла в секундах через ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    duration_str = subprocess.check_output(cmd).decode().strip()
    return math.ceil(float(duration_str))

def transcribe_large_file_in_chunks(path):
    """Режет файл на куски фиксированной длины и распознаёт каждый кусок."""
    total_duration = get_audio_duration(path)
    whole_text = ""

    for start in range(0, total_duration, CHUNK_LENGTH_SEC):
        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk_file.close()
        cmd = [
            "ffmpeg", "-y", "-i", path, "-ss", str(start),
            "-t", str(CHUNK_LENGTH_SEC), chunk_file.name
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        try:
            text = transcribe_with_whisper(chunk_file.name)
            whole_text += text.strip() + " "
        except Exception:
            continue
        finally:
            os.unlink(chunk_file.name)

    return whole_text.strip() or "(не удалось распознать речь)"

# ========== Streamlit UI ==========
st.title("🎙 Распознавание аудио в текст (Whisper)")
st.write("Загрузи аудиофайл (MP3, WAV, M4A, OGG, FLAC) и получи текст. Работает оффлайн.")

uploaded_file = st.file_uploader("Выбери аудиофайл", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    with st.spinner("⏳ Обработка..."):
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        temp_in.write(uploaded_file.read())
        temp_in.close()

        wav_path = convert_to_wav(temp_in.name)

        file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        if file_size_mb > 10:
            text_result = transcribe_large_file_in_chunks(wav_path)
        else:
            text_result = transcribe_with_whisper(wav_path)

        os.unlink(temp_in.name)
        os.unlink(wav_path)

    if text_result:
        st.success("✅ Распознавание завершено")
        st.text_area("Результат", text_result, height=300)
        st.download_button("💾 Скачать результат", text_result, file_name="transcription.txt")
