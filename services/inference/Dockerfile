FROM python:3.12.2

WORKDIR /src

COPY ./requirements.txt /src
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /src

CMD ["python", "-u", "src/inference.py"]