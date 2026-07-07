FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Mock mode doesn't need torch/transformers; real mode does. Install everything
# here since a production image should be able to run either mode.
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
