import os
import time
import http.client
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment, silence

# --- Если нужен офлайн-режим ---
try:
    from vosk import Model, KaldiRecognizer
    import wave
    import json
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# ======== Настройки ========
VOSK_MODEL_PATH = "vosk-model-ru"  # папка с моделью Vosk
MAX_RETRIES = 3  # количество попыток при ошибке API
# ===========================

def convert_to_wav(input_path):
    """Конвертирует любой аудиофайл в WAV (16kHz, mono)"""
    audio = AudioSegment.from_file(input_path)
    wav_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_large_file_on_silence(input_path, use_offline=False):
    """
    Распознаёт длинное аудио, режет по тишине, делает повтор запроса при ошибках.
    """
    # Конвертация в WAV при необходимости
    if not input_path.lower().endswith(".wav"):
        input_path = convert_to_wav(input_path)

    audio = AudioSegment.from_wav(input_path)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=-40,
        keep_silence=300
    )

    result_text = ""

    if use_offline:
        if not VOSK_AVAILABLE:
            st.error("Vosk не установлен. Поставьте пакет vosk для офлайн-распознавания.")
            return ""
        if not os.path.exists(VOSK_MODEL_PATH):
            st.error(f"Модель Vosk не найдена по пути {VOSK_MODEL_PATH}")
            return ""

        model = Model(VOSK_MODEL_PATH)

        for i, chunk in enumerate(chunks):
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            wf = wave.open(chunk_path, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())

            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    result_text += json.loads(rec.Result())["text"] + " "
            result_text += json.loads(rec.FinalResult())["text"] + " "

            wf.close()
            os.remove(chunk_path)

    else:
        recognizer = sr.Recognizer()

        for i, chunk in enumerate(chunks):
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)

                attempt = 0
                while attempt < MAX_RETRIES:
                    try:
                        text = recognizer.recognize_google(audio_data, language="ru-RU")
                        result_text += text + " "
                        break
                    except sr.UnknownValueError:
                        break
                    except (sr.RequestError, http.client.IncompleteRead) as e:
                        attempt += 1
                        time.sleep(1)

            os.remove(chunk_path)

    return result_text.strip()


# ========== Streamlit UI ==========
st.title("🎙 Распознавание аудио в текст")
st.write("Загрузи аудиофайл (MP3, WAV, M4A, OGG) и получи текст.")

use_offline = st.checkbox("Использовать офлайн-распознавание (Vosk)", value=False)

uploaded_file = st.file_uploader("Выбери аудиофайл", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    # Сохраняем во временный файл
    input_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("⏳ Обработка...")
    text_result = transcribe_large_file_on_silence(input_path, use_offline=use_offline)

    if text_result:
        st.success("✅ Распознавание завершено")
        st.text_area("Результат", text_result, height=300)
        st.download_button("💾 Скачать результат", text_result, file_name="transcription.txt")

    # Чистим временные файлы
    os.remove(input_path)
