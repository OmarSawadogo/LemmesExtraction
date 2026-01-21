"""
Configuration centralisée du système d'extraction de connaissances.
Charge les paramètres depuis les variables d'environnement ou utilise les valeurs par défaut.
"""

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Charger le fichier .env si présent
load_dotenv()


class Config:
    """Classe de configuration centralisée."""

    # Répertoire racine du projet
    BASE_DIR = Path(__file__).parent.parent

    # Configuration Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    LLAVA_MODEL: str = os.getenv("LLAVA_MODEL", "llava:13b")

    # Chemins de données
    ONTOLOGY_DIR: Path = BASE_DIR / "data" / "ontology"
    ONTOLOGY_PATH: str = os.getenv(
        "ONTOLOGY_PATH",
        str(BASE_DIR / "data" / "ontology" / "ontologie_plantes_burkina_faso.ttl")
    )
    IMAGES_PATH: str = os.getenv(
        "IMAGES_PATH",
        str(BASE_DIR / "data" / "images")
    )
    EXPORT_PATH: str = os.getenv(
        "EXPORT_PATH",
        str(BASE_DIR / "exports")
    )

    # Seuils de similarité par défaut
    THRESHOLD_ENTITIES: float = float(os.getenv("THRESHOLD_ENTITIES", "0.75"))
    THRESHOLD_RELATIONS: float = float(os.getenv("THRESHOLD_RELATIONS", "0.70"))
    THRESHOLD_ATTRIBUTES: float = float(os.getenv("THRESHOLD_ATTRIBUTES", "0.65"))

    # Configuration Gradio
    GRADIO_SERVER_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SERVER_NAME: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    # Modèle d'embeddings (Ollama)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "nextfire/paraphrase-multilingual-minilm:l12-v2"
    )

    # Algorithme de similarité par défaut
    SIMILARITY_ALGORITHM: str = os.getenv("SIMILARITY_ALGORITHM", "hybrid")

    @classmethod
    def get_available_ontologies(cls) -> Dict[str, str]:
        """
        Retourne les ontologies disponibles dans le dossier data/ontology.

        Returns:
            Dict[str, str]: Dictionnaire {nom_fichier: chemin_complet}
        """
        ontologies = {}
        if cls.ONTOLOGY_DIR.exists():
            for file_path in cls.ONTOLOGY_DIR.glob("*.ttl"):
                ontologies[file_path.name] = str(file_path)
        return ontologies

    @classmethod
    def get_thresholds(cls) -> Dict[str, float]:
        """Retourne les seuils de similarité sous forme de dictionnaire."""
        return {
            "entities": cls.THRESHOLD_ENTITIES,
            "relations": cls.THRESHOLD_RELATIONS,
            "attributes": cls.THRESHOLD_ATTRIBUTES
        }

    @classmethod
    def validate_paths(cls) -> bool:
        """
        Valide l'existence des chemins critiques.

        Returns:
            bool: True si tous les chemins existent, False sinon.
        """
        ontology_path = Path(cls.ONTOLOGY_PATH)
        images_path = Path(cls.IMAGES_PATH)

        if not ontology_path.exists():
            print(f"⚠️  Fichier ontologie introuvable: {ontology_path}")
            return False

        if not images_path.exists():
            print(f"⚠️  Dossier images introuvable: {images_path}")
            return False

        # Créer le dossier exports s'il n'existe pas
        export_path = Path(cls.EXPORT_PATH)
        export_path.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def display_config(cls):
        """Affiche la configuration actuelle."""
        print("=" * 60)
        print("📋 CONFIGURATION DU SYSTÈME")
        print("=" * 60)
        print(f"🔗 Ollama URL: {cls.OLLAMA_BASE_URL}")
        print(f"🤖 Modèle LLaVA: {cls.LLAVA_MODEL}")
        print(f"📚 Ontologie: {cls.ONTOLOGY_PATH}")
        print(f"🖼️  Images: {cls.IMAGES_PATH}")
        print(f"💾 Exports: {cls.EXPORT_PATH}")
        print(f"🎯 Seuils: Entités={cls.THRESHOLD_ENTITIES}, "
              f"Relations={cls.THRESHOLD_RELATIONS}, "
              f"Attributs={cls.THRESHOLD_ATTRIBUTES}")
        print(f"🌐 Gradio: {cls.GRADIO_SERVER_NAME}:{cls.GRADIO_SERVER_PORT}")
        print(f"🧠 Embeddings: {cls.EMBEDDING_MODEL}")
        print("=" * 60)


# Instance globale de configuration
config = Config()
