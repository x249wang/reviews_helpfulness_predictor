FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["waitress-serve", "--port=8080", "app:app"]

# docker build -t reviews_app .
# docker run -p 8080:8080 -d reviews_app
