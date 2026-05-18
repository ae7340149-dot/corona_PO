import sys
import os
import tempfile
import torch
import streamlit as st
from pathlib import Path
from typing import Optional

# --- ЖЕЛЕЗОБЕТОННЫЙ ХАК ПУТЕЙ ---
# Добавляем папку с кодом в пути поиска Python, чтобы избежать ошибок с названиями (src/scr)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "scr"))
sys.path.append(os.path.join(base_dir, "src"))

# Теперь используем прямые импорты (без приставок)
from preprocessing import AudioPreprocessor
from model import CoughCNN
# --------------------------------

# Настройка страницы (должна быть первой командой Streamlit)
st.set_page_config(
    page_title="Анализ кашля на COVID-19",
    layout="centered"
)

@st.cache_resource
def load_model(model_path: str = "models/best_cough_cnn.pth") -> Optional[CoughCNN]:
    """
    Загружает обученную модель и кэширует её в памяти.
    Декоратор @st.cache_resource предотвращает повторную загрузку весов 
    при каждом обновлении страницы, ускоряя работу интерфейса.
    """
    # Для веб-интерфейса безопаснее всего использовать CPU по умолчанию
    device = torch.device("cpu")
    
    model = CoughCNN()
    
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
        return model
    else:
        st.error(f"Файл модели не найден по пути: {model_path}. Сначала запустите обучение!")
        return None

def main() -> None:
    # Заголовок страницы
    st.title("Анализ кашля на COVID-19 (НИРС)")
    st.markdown("""
    Данная система использует сверточную нейронную сеть (CNN) для бинарной классификации 
    аудиозаписей кашля. Загрузите аудиофайл, чтобы система оценила вероятность 
    наличия паттернов, характерных для COVID-19.
    """)

    # Инициализация модели и препроцессора
    model = load_model()
    preprocessor = AudioPreprocessor()
    
    # Порог для бинарной классификации
    decision_threshold = 0.5

    # Виджет загрузки файлов (расширен список форматов)
    uploaded_file = st.file_uploader(
        "Загрузите аудиофайл с записью кашля", 
        type=["wav", "mp3", "webm", "ogg", "m4a"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        if st.button("Анализировать", use_container_width=True):
            if model is None:
                st.error("Модель не загружена. Анализ невозможен.")
                return

            with st.spinner('Обработка аудио и анализ нейросетью...'):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # 1. Предобработка
                    raw_tensor = preprocessor.process(tmp_file_path)
                    
                    if raw_tensor is None:
                        st.error("Не удалось прочитать аудио. Возможно, файл поврежден.")
                    else:
                        # Добавляем размерность батча
                        input_tensor = raw_tensor.unsqueeze(0)

                        # 2. Инференс
                        with torch.no_grad():
                            logits = model(input_tensor)
                            probability = torch.sigmoid(logits).item() 

                        # 3. Вывод результата
                        percent = probability * 100
                        
                        st.markdown("---")
                        st.subheader("Результат анализа:")

                        if probability < decision_threshold:
                            st.success("Здоров. Паттерны COVID-19 не обнаружены.")
                            st.info(f"Вероятность заболевания: {percent:.2f}%")
                        else:
                            st.error("Внимание! Обнаружен паттерн COVID-19.")
                            st.warning(f"Вероятность заболевания: {percent:.2f}%")
                            
                        st.caption("Примечание: Результат работы нейросети не является медицинским диагнозом.")

                except Exception as e:
                    st.error(f"Произошла ошибка при обработке файла: {e}")
                finally:
                    # Удаляем временный файл в блоке finally, чтобы он удалялся всегда
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()