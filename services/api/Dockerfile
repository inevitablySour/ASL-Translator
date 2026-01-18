FROM python:3.12.2

WORKDIR /src

COPY ./requirements.txt /src
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /src

# Make entrypoint executable
RUN chmod +x /src/entrypoint.sh

CMD ["/src/entrypoint.sh"]
