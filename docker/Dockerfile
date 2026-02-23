FROM python:3.12
RUN mkdir -p /experiment
RUN pip install poetry==2.1.3
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libfreetype6 \
    libpng16-16 \
    fonts-dejavu-core \
    libgl1 \
    libglu1-mesa \
    libgl1-mesa-dri \
    mesa-utils \
    xvfb \
    xauth \
 && rm -rf /var/lib/apt/lists/*
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
WORKDIR /experiment
COPY . /experiment
RUN poetry install
ENV OWNER=1000:1000
CMD xvfb-run -s "-screen 0 1400x900x24" poetry run python3 $TASK