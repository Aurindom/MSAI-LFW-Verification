FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir facenet-pytorch --no-deps && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')" \
    && echo "FaceNet weights cached."

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/verify.py"]
CMD ["--help"]
