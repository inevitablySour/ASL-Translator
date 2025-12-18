FROM python:3.12.2

WORKDIR /src

COPY ./requirements.txt /src
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /src

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]