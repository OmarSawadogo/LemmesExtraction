FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Lemmes Extraction System"
LABEL description="Système d'extraction de connaissances de plantes agricoles du Burkina Faso"

# Variables d'environnement Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY data/ ./data/

# Création du dossier exports
RUN mkdir -p /app/exports

# Exposition du port Gradio
EXPOSE 7860

# Commande par défaut
CMD ["python", "-m", "src.app"]
