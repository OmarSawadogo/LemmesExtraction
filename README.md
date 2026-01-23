# Systeme d'Extraction de Connaissances de Plantes Agricoles

Systeme d'extraction automatique de connaissances a partir d'images de plantes agricoles du Burkina Faso (mais, oignon, tomate). Combine des LLM locaux via Ollama (LLaVA, Qwen2.5-VL, Llama3.2-Vision) et une ontologie de domaine pour produire un schema structure Data Vault.

## Fonctionnalites

- **Analyse d'images** de plantes avec modeles de vision (LLaVA, Qwen, Llama)
- **Extraction automatique** de lemmes descriptifs
- **Classification ontologique** guidee (Hubs, Links, Satellites)
- **Calcul de similarite** hybride (lexicale + semantique)
- **Schema Data Vault** structure et valide
- **Exports multi-formats** : JSON, RDF/Turtle, SQL
- **Interface Gradio** interactive
- **Execution locale** avec Docker et Ollama

## Architecture

```
Image de plante
      |
      v
+-----------------------------------+
| ETAPE 1: Extraction LLM (LLaVA)   |
| Lemmes: [mais, malade, necrose]   |
+-----------------------------------+
      |
      v
+-----------------------------------+
| ETAPE 2: Classification           |
| REGLE 1: Hubs (Entites)           |
| REGLE 2: Links (Relations)        |
| REGLE 3: Satellites (Attributs)   |
+-----------------------------------+
      |
      v
+-----------------------------------+
| ETAPE 3: Calcul de Similarite     |
| - Lexicale (Jaro-Winkler)         |
| - Semantique (Embeddings)         |
+-----------------------------------+
      |
      v
+-----------------------------------+
| ETAPE 4: Schema Data Vault        |
| - Hubs, Links, Satellites         |
| - Validation, Metadonnees         |
+-----------------------------------+
      |
      v
  JSON  RDF  SQL
```

## Prerequis

- **Docker** et **Docker Compose** (recommande)
- **Python 3.11** (si installation locale)
- **4-10 GB RAM** selon le modele (4GB pour llava:7b, 10GB pour llava:13b)
- **GPU NVIDIA** (optionnel)

## Installation

### Option 1: Docker (Recommande)

```bash
# 1. Cloner le repository
git clone <votre-repo>
cd LemmesExtraction

# 2. Copier et adapter le fichier d'environnement
cp .env.example .env

# 3. Lancer les services
docker-compose up --build -d

# 4. Telecharger les modeles LLM
docker-compose exec ollama ollama pull llava:7b          # 4-5 GB RAM
docker-compose exec ollama ollama pull qwen2.5vl:latest  # 6 GB RAM
docker-compose exec ollama ollama pull llama3.2-vision:latest  # 7-8 GB RAM

# 5. Verifier les logs
docker-compose logs -f app
```

L'interface Gradio sera accessible sur **http://localhost:7860**

### Option 2: Installation locale

```bash
# 1. Creer un environnement virtuel Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Installer les dependances
pip install -r requirements.txt

# 3. Installer Ollama separement (https://ollama.com/download)

# 4. Telecharger les modeles LLM
ollama pull llava:7b

# 5. Copier et adapter la configuration
cp .env.example .env
# Editer .env: OLLAMA_BASE_URL=http://localhost:11434

# 6. Lancer Ollama (terminal separe)
ollama serve

# 7. Lancer l'application
python main.py
```

## Utilisation

### Interface Gradio

1. Ouvrir http://localhost:7860
2. **Onglet Analyse** : Uploader une image, choisir le modele, analyser
3. **Onglet Traitement par lot** : Traiter plusieurs images
4. **Onglet Export** : Exporter en JSON, RDF ou SQL
5. **Onglet Ontologie** : Consulter les statistiques

### Commandes Docker utiles

```bash
docker-compose logs -f app          # Voir les logs
docker-compose restart              # Redemarrer
docker-compose down                 # Arreter
docker-compose exec ollama ollama list  # Lister les modeles
```

## Structure du projet

```
LemmesExtraction/
├── docker-compose.yml          # Orchestration Docker
├── Dockerfile                  # Image Python
├── requirements.txt            # Dependances Python
├── .env.example                # Template de configuration
├── README.md                   # Documentation
├── main.py                     # Point d'entree
├── data/
│   ├── images/                 # Images d'entree
│   └── ontology/
│       └── ontologie_plantes_burkina_faso.ttl
├── exports/                    # Fichiers exportes
└── src/
    ├── __init__.py
    ├── app.py                  # Interface Gradio
    ├── config.py               # Configuration
    ├── llm_extractor.py        # Extraction LLaVA
    ├── ontology_loader.py      # Chargement ontologie RDF
    ├── similarity_calculator.py # Calcul similarite
    ├── ontology_matcher.py     # Classification
    ├── datavault_generator.py  # Generation schema
    ├── models/
    │   ├── hub.py              # Modele Hub
    │   ├── link.py             # Modele Link
    │   └── satellite.py        # Modele Satellite
    └── exporters/
        ├── json_exporter.py    # Export JSON
        ├── rdf_exporter.py     # Export RDF/Turtle
        └── sql_exporter.py     # Export SQL
```

## Configuration

### Variables d'environnement (.env)

```bash
# Ollama
OLLAMA_BASE_URL=http://ollama:11434
LLAVA_MODEL=llava:7b

# Chemins
ONTOLOGY_PATH=data/ontology/ontologie_plantes_burkina_faso.ttl
IMAGES_PATH=data/images/
EXPORT_PATH=exports/

# Seuils de similarite
THRESHOLD_ENTITIES=0.75
THRESHOLD_RELATIONS=0.70
THRESHOLD_ATTRIBUTES=0.65

# Gradio
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0

# Embeddings
EMBEDDING_MODEL=nextfire/paraphrase-multilingual-minilm:l12-v2

# Algorithme de similarite
SIMILARITY_ALGORITHM=hybrid
```

## Comparaison des modeles

| Modele | RAM | Vitesse | Precision | Usage |
|--------|-----|---------|-----------|-------|
| llava:7b | 4-5 GB | Rapide | Bonne | Tests, machines limitees |
| qwen2.5vl:latest | 6 GB | Tres rapide | Tres bonne | Recommande |
| llama3.2-vision | 7-8 GB | Rapide | Tres bonne | Usage general |
| llava:13b | 10 GB | Moyen | Excellente | Production |

## Depannage

### Ollama non disponible

```bash
docker-compose ps ollama
docker-compose logs ollama
docker-compose restart ollama
```

### Modele introuvable

```bash
docker-compose exec ollama ollama pull llava:7b
docker-compose exec ollama ollama list
```

### Memoire insuffisante

- Selectionner `llava:7b` dans l'interface (4GB)
- Augmenter la RAM Docker dans Docker Desktop

## Technologies

- **LLM/Vision:** Ollama (LLaVA, Qwen, Llama Vision)
- **Ontologie:** RDFLib
- **Similarite:** Jellyfish (Jaro-Winkler), Embeddings Ollama
- **Interface:** Gradio
- **Containerisation:** Docker
- **Langage:** Python 3.11

## License

MIT License

---

**Version:** 1.0.0
