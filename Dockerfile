FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]  # Use FastAPI if needed
