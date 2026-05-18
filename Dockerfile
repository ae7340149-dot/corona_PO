# Используем легковесный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# МЕНЯЕМ СЕРВЕРЫ DEBIAN НА ЯНДЕКС (чтобы качалось быстро и без ошибок)
RUN sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources || true && \
    sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list || true

# Устанавливаем системные зависимости для аудио
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# УВЕЛИЧИВАЕМ ТАЙМАУТ И КАЧАЕМ ЛЕГКУЮ ВЕРСИЮ PYTORCH (CPU)
# КАЧАЕМ БИБЛИОТЕКИ ЧЕРЕЗ КИТАЙСКИЕ ЗЕРКАЛА ALIBABA (обход обрыва SSL)
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Копируем весь исходный код проекта
COPY . .

# Сообщаем Python, что корень проекта — это папка /app
ENV PYTHONPATH="/app"

# Открываем порт для Streamlit
EXPOSE 8501

# Запуск приложения
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]