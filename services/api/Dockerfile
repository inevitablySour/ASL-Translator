FROM python:3.12.2

WORKDIR /src

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /src
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /src

# Make entrypoint executable
RUN chmod +x /src/entrypoint.sh

CMD ["/src/entrypoint.sh"]
