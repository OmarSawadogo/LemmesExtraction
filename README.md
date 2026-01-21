# 🌾 Système d'Extraction de Connaissances de Plantes Agricoles

Système d'extraction automatique de connaissances à partir d'images de plantes agricoles du Burkina Faso (maïs, oignon, tomate). Combine des LLM locaux via Ollama (LLaVA 13b) et une ontologie de domaine pour produire un schéma structuré Data Vault.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline d'extraction](#-pipeline-dextraction)
- [Structure du projet](#-structure-du-projet)
- [Configuration](#-configuration)
- [API](#-api)
- [Dépannage](#-dépannage)

## ✨ Fonctionnalités

- **Analyse d'images** de plantes avec LLaVA (vision + langage)
- **Extraction automatique** de lemmes descriptifs
- **Classification ontologique** guidée (Hubs, Links, Satellites)
- **Calcul de similarité** hybride (lexicale + sémantique)
- **Schéma Data Vault** structuré et validé
- **Exports multi-formats** : JSON, RDF/Turtle, SQL
- **Interface Gradio** interactive et intuitive
- **Exécution locale** avec Docker et Ollama

## 🏗️ Architecture

```
┌─────────────────┐
│  Image de       │
│  plante         │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ ÉTAPE 1: Extraction LLM (LLaVA)     │
│ Lemmes: [mais, malade, necrose, ...] │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ ÉTAPE 2: Classification Ontologique │
│ RÈGLE 1: Hubs (Entités)             │
│ RÈGLE 2: Links (Relations)          │
│ RÈGLE 3: Satellites (Attributs)     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ ÉTAPE 3: Calcul de Similarité       │
│ - Lexicale (Jaro-Winkler)           │
│ - Sémantique (Embeddings)           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ ÉTAPE 4: Schéma Data Vault          │
│ - Hubs, Links, Satellites           │
│ - Validation, Métadonnées           │
└────────┬────────────────────────────┘
         │
         ▼
   ┌────┴────┐
   │ Exports │
   └─┬───┬───┬┘
     │   │   │
   JSON RDF SQL
```

## 📦 Prérequis

- **Docker** et **Docker Compose** (recommandé)
- **Python 3.11** (si installation locale)
- **4-10 GB RAM** (selon le modèle choisi: 4GB pour llava:7b, 10GB pour llava:13b)
- **GPU NVIDIA** (optionnel, pour accélération)

## 🚀 Installation

### Option 1: Docker (Recommandé)

```bash
# 1. Cloner le repository
git clone <votre-repo>
cd LemmesExtraction

# 2. Copier et adapter le fichier d'environnement
cp .env.example .env

# 3. Lancer les services
docker-compose up --build -d

# 4. Télécharger les modèles LLM (première utilisation)
# Choisissez selon votre RAM disponible:
docker-compose exec ollama ollama pull llava:7b          # Recommandé: 4-5 GB RAM
docker-compose exec ollama ollama pull qwen2.5vl:latest  # Moderne: 6 GB RAM
docker-compose exec ollama ollama pull llama3.2-vision:latest  # Optionnel: 7-8 GB RAM
docker-compose exec ollama ollama pull llava:13b         # Optionnel: 10 GB RAM (⚠️ Haute RAM)

# 5. Vérifier les logs
docker-compose logs -f app
```

L'interface Gradio sera accessible sur **http://localhost:7860**

### Option 2: Installation locale

```bash
# 1. Créer un environnement virtuel Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Installer Ollama séparément
# https://ollama.com/download

# 4. Télécharger les modèles LLM (au choix)
ollama pull llava:7b          # Recommandé: 4-5 GB RAM
ollama pull qwen2.5vl:latest  # Moderne: 6 GB RAM
ollama pull llama3.2-vision:latest  # Optionnel: 7-8 GB RAM
ollama pull llava:13b         # Optionnel: 10 GB RAM (⚠️ Haute RAM)

# 5. Copier et adapter la configuration
cp .env.example .env
# Éditer .env pour mettre OLLAMA_BASE_URL=http://localhost:11434

# 6. Lancer Ollama (dans un terminal séparé)
ollama serve

# 7. Lancer l'application
python src/app.py
```

## 💻 Utilisation

### Interface Gradio

1. **Ouvrir** http://localhost:7860
2. **Onglet "Analyse d'Image"** :
   - Uploader une image de plante
   - Choisir le modèle LLM (llava:7b, qwen2.5vl, llama3.2-vision, ou llava:13b)
   - Ajuster les seuils de similarité (optionnel)
   - Cliquer sur "Analyser"
   - Visualiser les résultats (Hubs, Links, Satellites)
3. **Onglet "Export"** :
   - Choisir le format (JSON, RDF, SQL)
   - Cliquer sur "Exporter"
   - Télécharger le fichier généré
4. **Onglet "Ontologie"** :
   - Consulter les statistiques de l'ontologie

### Commandes Docker utiles

```bash
# Voir les logs
docker-compose logs -f app
docker-compose logs -f ollama

# Redémarrer les services
docker-compose restart

# Arrêter les services
docker-compose down

# Reconstruire après modification du code
docker-compose up --build -d

# Vérifier les modèles Ollama disponibles
docker-compose exec ollama ollama list
```

## 🔄 Pipeline d'extraction

### ÉTAPE 1: Extraction de lemmes (LLaVA)

```python
# Entrée: Image de plante
# Sortie: Liste de lemmes
lemmes = ["mais", "malade", "helminthosporiose", "necrose",
          "vert_moyen", "beige_brun", "lineaire_lanceolee"]
```

### ÉTAPE 2: Classification ontologique

**RÈGLE 1: Hubs (Entités)**
- Seuil: θe = 0.75
- Détecte: plantes, maladies, symptômes
- Exemple: "mais" → Hub(type=plante), "helminthosporiose" → Hub(type=maladie)

**RÈGLE 2: Links (Relations)**
- Seuil: θr = 0.70
- Détecte: relations entre entités
- Exemple: Link(mais → a_maladie → helminthosporiose)

**RÈGLE 3: Satellites (Attributs)**
- Seuil: θa = 0.65
- Détecte: attributs descriptifs
- Exemple: Satellite(hub=mais, attribut=couleur_feuille, valeur=vert_moyen)

### ÉTAPE 3: Calcul de similarité

```python
sim(lemme, concept) = max(sim_lex, sim_sem)

# Similarité lexicale (Jaro-Winkler)
sim_lex("mais", "maïs") = 1.0

# Similarité sémantique (Embeddings)
sim_sem("jaunissement", "chlorose") = 0.82
```

### ÉTAPE 4: Schéma Data Vault

```json
{
  "hubs": [
    {
      "hub_key": "abc123...",
      "business_key": "mais",
      "entity_type": "plante",
      "confidence_score": 0.98
    }
  ],
  "links": [...],
  "satellites": [...]
}
```

## 📁 Structure du projet

```
LemmesExtraction/
├── docker-compose.yml          # Orchestration Docker
├── Dockerfile                  # Image Python
├── requirements.txt            # Dépendances Python
├── .env.example               # Template de configuration
├── README.md                  # Documentation
├── data/
│   ├── images/                # Images d'entrée
│   │   ├── img1.jpg          # Maïs sain
│   │   ├── img2.jpg          # Tomate avec acariens
│   │   ├── img3.jpg          # Maïs avec chenille
│   │   └── img4.jpg          # Maïs avec helminthosporiose
│   └── ontology/
│       └── ontologie_plantes_burkina_faso.ttl  # Ontologie RDF
├── exports/                   # Fichiers exportés (généré)
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration centralisée
│   ├── app.py                 # Interface Gradio (point d'entrée)
│   ├── llm_extractor.py       # Extraction LLaVA
│   ├── ontology_loader.py     # Chargement ontologie RDF
│   ├── similarity_calculator.py # Calcul similarité
│   ├── ontology_matcher.py    # Classification (3 règles)
│   ├── datavault_generator.py # Génération schéma Data Vault
│   ├── models/
│   │   ├── hub.py             # Modèle Hub
│   │   ├── link.py            # Modèle Link
│   │   └── satellite.py       # Modèle Satellite
│   └── exporters/
│       ├── json_exporter.py   # Export JSON
│       ├── rdf_exporter.py    # Export RDF/Turtle
│       └── sql_exporter.py    # Export SQL
└── tests/
    └── test_pipeline.py       # Tests d'intégration
```

## ⚙️ Configuration

### Variables d'environnement (.env)

```bash
# Ollama
OLLAMA_BASE_URL=http://ollama:11434
LLAVA_MODEL=llava:13b

# Chemins
ONTOLOGY_PATH=data/ontology/ontologie_plantes_burkina_faso.ttl
IMAGES_PATH=data/images/
EXPORT_PATH=exports/

# Seuils de similarité
THRESHOLD_ENTITIES=0.75    # Seuil pour Hubs
THRESHOLD_RELATIONS=0.70   # Seuil pour Links
THRESHOLD_ATTRIBUTES=0.65  # Seuil pour Satellites

# Gradio
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0

# Embeddings
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

### Ajustement des seuils

- **Seuils élevés (0.8-1.0)** : Précision maximale, peut manquer des correspondances
- **Seuils moyens (0.65-0.75)** : Bon équilibre (recommandé)
- **Seuils faibles (0.5-0.65)** : Capture plus de correspondances, moins précis

Les seuils peuvent être ajustés dynamiquement via l'interface Gradio.

## 🔌 API

### Utilisation programmatique

```python
from src.ontology_loader import OntologyLoader
from src.llm_extractor import LLMExtractor
from src.similarity_calculator import SimilarityCalculator
from src.ontology_matcher import OntologyMatcher
from src.datavault_generator import DataVaultGenerator

# 1. Charger l'ontologie
ontology = OntologyLoader("data/ontology/ontologie_plantes_burkina_faso.ttl")

# 2. Initialiser les composants
llm_extractor = LLMExtractor("http://localhost:11434", "llava:13b")
similarity_calc = SimilarityCalculator()
matcher = OntologyMatcher(ontology, similarity_calc, {
   "entities": 0.75,
   "relations": 0.70,
   "attributes": 0.65
})

# 3. Analyser une image
lemmas = llm_extractor.extract_lemmas("data/images/img1.jpg")
hubs, links, satellites = matcher.classify_lemmas(lemmas, "img1.jpg")

# 4. Générer le schéma
generator = DataVaultGenerator()
schema = generator.generate_schema(hubs, links, satellites, "img1.jpg", lemmas)

# 5. Exporter
from src.exporters.json_exporter import JSONExporter

exporter = JSONExporter()
exporter.export(schema, "exports/schema.json")
```

## 🐛 Dépannage

### Erreur: Ollama non disponible

```bash
# Vérifier que le service Ollama est lancé
docker-compose ps ollama

# Vérifier les logs
docker-compose logs ollama

# Redémarrer Ollama
docker-compose restart ollama
```

### Erreur: Modèle LLaVA introuvable

```bash
# Télécharger le modèle
docker-compose exec ollama ollama pull llava:13b

# Vérifier les modèles disponibles
docker-compose exec ollama ollama list
```

### Erreur: Out of memory

- **Réduire la taille du modèle** : Dans l'interface, sélectionner `llava:7b` (4GB) au lieu de `llava:13b` (10GB)
- **Augmenter la mémoire Docker** : Dans Docker Desktop, augmenter la RAM allouée
- **Utiliser un GPU** : Décommenter les sections GPU dans docker-compose.yml (Linux/WSL2 + NVIDIA)

### Comparaison des modèles

| Modèle | RAM requise | Vitesse | Précision | Usage recommandé |
|--------|-------------|---------|-----------|------------------|
| llava:7b | 4-5 GB | Rapide | Bonne | Tests, machines limitées |
| qwen2.5vl:latest | 6 GB | Très rapide | Très bonne | Moderne, multilingue performant ⭐ |
| llama3.2-vision | 7-8 GB | Rapide | Très bonne | Usage général, équilibré |
| llava:13b | 10 GB | Moyen | Excellente | Production, précision max (⚠️ RAM élevée) |

### Erreur: Ontologie introuvable

```bash
# Vérifier que le fichier existe
ls data/ontology/ontologie_plantes_burkina_faso.ttl

# Vérifier les chemins dans .env
cat .env | grep ONTOLOGY_PATH
```

### Interface Gradio ne se charge pas

```bash
# Vérifier les logs de l'application
docker-compose logs app

# Vérifier que le port 7860 n'est pas déjà utilisé
netstat -an | grep 7860  # Linux/Mac
netstat -an | findstr 7860  # Windows

# Redémarrer l'application
docker-compose restart app
```

## 📊 Exemples de résultats

### Exemple 1: Maïs sain (img1.jpg)

**Lemmes extraits:**
```
mais, sain, vert_fonce, lineaire_lanceolee, nervation_parallele, lisse
```

**Hubs:**
- `mais` (plante, score: 1.0)

**Satellites:**
- `couleur_feuille` = vert_fonce (score: 0.92)
- `forme_feuille` = lineaire_lanceolee (score: 0.98)
- `nervation` = nervation_parallele (score: 0.95)

### Exemple 2: Maïs avec helminthosporiose (img4.jpg)

**Lemmes extraits:**
```
mais, malade, helminthosporiose, necrose, vert_moyen, beige_brun
```

**Hubs:**
- `mais` (plante, score: 1.0)
- `helminthosporiose` (maladie, score: 0.96)
- `necrose` (symptome, score: 0.89)

**Links:**
- `mais` → `a_maladie_mais` → `helminthosporiose` (score: 0.85)
- `mais` → `presente_symptome` → `necrose` (score: 0.80)
- `helminthosporiose` → `cause_symptome` → `necrose` (score: 0.75)

**Satellites:**
- `couleur_feuille` = vert_moyen (score: 0.88)
- `couleur_feuille` = beige_brun (score: 0.91)

## 🔬 Technologies

- **LLM/Vision:** Ollama (LLaVA 7b/13b, Qwen 2.5 Vision, Llama 3.2 Vision)
- **Ontologie:** RDFLib, OWL/RDF
- **Similarité:** Jellyfish (Jaro-Winkler), Sentence-Transformers
- **Interface:** Gradio
- **Containerisation:** Docker, Docker Compose
- **Langage:** Python 3.11

## 📝 License

MIT License

## 👥 Contributeurs

Développé pour l'analyse de plantes agricoles du Burkina Faso.

## 📧 Contact

Pour toute question ou contribution, ouvrir une issue sur le repository.

---

**Version:** 1.0.0
**Dernière mise à jour:** 2026-01-07
