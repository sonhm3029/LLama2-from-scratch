FROM python:3.9.13-slim-buster

ENV PYTHONUNBUFFERED=1

WORKDIR /src

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]