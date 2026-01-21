"""
Application Gradio pour l'extraction de connaissances de plantes agricoles.
Interface web pour analyser les images, visualiser les résultats et exporter les données.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import time
import os
import shutil
from glob import glob

# Imports des modules du projet
from src.config import config
from src.ontology_loader import OntologyLoader
from src.llm_extractor import LLMExtractor
from src.similarity_calculator import SimilarityCalculator
from src.ontology_matcher import OntologyMatcher
from src.datavault_generator import DataVaultGenerator, DataVaultSchema
from src.exporters.json_exporter import JSONExporter
from src.exporters.rdf_exporter import RDFExporter
from src.exporters.sql_exporter import SQLExporter


# ========== INITIALISATION GLOBALE ==========

print("\n" + "=" * 60)
print("🌾 SYSTÈME D'EXTRACTION DE CONNAISSANCES DE PLANTES")
print("=" * 60)

# Valider les chemins
if not config.validate_paths():
    print("❌ Erreur: Fichiers requis manquants. Vérifiez la configuration.")
    exit(1)

config.display_config()

# Charger l'ontologie par défaut (sera rechargée dynamiquement selon la sélection)
print("\n📚 Chargement de l'ontologie...")
ontology = OntologyLoader(config.ONTOLOGY_PATH)
ontology_stats = ontology.get_statistics()
print(f"✅ Ontologie chargée avec succès")

# Initialiser le calculateur de similarité par défaut (sera recréé avec les paramètres sélectionnés)
print("\n🧠 Initialisation du calculateur de similarité...")
similarity_calc = SimilarityCalculator(
    embedding_model=config.EMBEDDING_MODEL,
    ollama_base_url=config.OLLAMA_BASE_URL,
    algorithm=config.SIMILARITY_ALGORITHM
)

# Initialiser l'extracteur LLM
print("\n🤖 Connexion à Ollama...")
try:
    llm_extractor = LLMExtractor(config.OLLAMA_BASE_URL, config.LLAVA_MODEL)
    if not llm_extractor.check_model_availability():
        print("⚠️  Le modèle LLaVA n'est pas disponible. Certaines fonctionnalités seront limitées.")
except Exception as e:
    print(f"⚠️  Impossible de se connecter à Ollama: {e}")
    llm_extractor = None

# Variables globales pour stocker les schémas générés
last_schema: Optional[DataVaultSchema] = None
batch_schemas: Dict[str, DataVaultSchema] = {}


# ========== FONCTIONS DE L'INTERFACE ==========

def clear_outputs():
    """Affiche un message de chargement avant l'analyse."""
    status_msg = "⏳ **ANALYSE EN COURS**\n\nVeuillez patienter, le modèle traite votre image..."
    return status_msg, "", "", ""


def analyze_image(
    image,
    selected_model: str,
    selected_ontology: str,
    selected_similarity: str,
    threshold_entities: float,
    threshold_relations: float,
    threshold_attributes: float,
    progress=gr.Progress()
) -> Tuple[str, str, str, str]:
    """
    Analyse une image et retourne les résultats organisés par Hub.

    Args:
        image: Image uploadée
        selected_model: Modèle LLM à utiliser
        selected_ontology: Fichier d'ontologie à utiliser
        selected_similarity: Algorithme de similarité à utiliser
        threshold_entities: Seuil pour entités
        threshold_relations: Seuil pour relations
        threshold_attributes: Seuil pour attributs
        progress: Gestionnaire de progression Gradio

    Returns:
        Tuple (status_msg, lemmes_str, results_by_hub_str, stats_str)
    """
    global last_schema

    # Initialiser la progression immédiatement
    progress(0, desc="Démarrage de l'analyse...")

    if image is None:
        return "❌ Aucune image fournie", "", "", ""

    try:
        start_time = time.time()

        # ÉTAPE 1: Extraction de lemmes avec le modèle sélectionné
        progress(0.05, desc="Connexion au modèle LLM...")
        print(f"\n🔍 Analyse de l'image avec le modèle {selected_model}...")
        current_extractor = LLMExtractor(config.OLLAMA_BASE_URL, selected_model)

        # Vérifier la disponibilité (mais continuer même si la vérification échoue)
        progress(0.1, desc="Vérification du modèle...")
        is_available = current_extractor.check_model_availability()
        if not is_available:
            print(f"⚠️  La vérification du modèle a échoué, mais on essaie quand même...")

        progress(0.2, desc="Extraction des lemmes...")
        lemmas = current_extractor.extract_lemmas(image)

        if not lemmas:
            return "⚠️  Aucun lemme extrait", "", "", ""

        progress(0.4, desc=f"{len(lemmas)} lemmes extraits, chargement ontologie...")

        # ÉTAPE 2: Charger l'ontologie sélectionnée
        available_ontologies = config.get_available_ontologies()
        ontology_path = available_ontologies.get(selected_ontology, config.ONTOLOGY_PATH)
        current_ontology = OntologyLoader(ontology_path)
        print(f"📚 Ontologie chargée: {selected_ontology}")

        # ÉTAPE 3: Initialiser le calculateur de similarité avec l'algorithme sélectionné
        progress(0.45, desc="Initialisation du calculateur de similarité...")
        current_similarity_calc = SimilarityCalculator(
            embedding_model=config.EMBEDDING_MODEL,
            ollama_base_url=config.OLLAMA_BASE_URL,
            algorithm=selected_similarity
        )
        print(f"📊 Algorithme de similarité: {selected_similarity}")

        # ÉTAPE 4: Classification ontologique
        progress(0.5, desc="Classification ontologique...")
        thresholds = {
            "entities": threshold_entities,
            "relations": threshold_relations,
            "attributes": threshold_attributes
        }

        matcher = OntologyMatcher(current_ontology, current_similarity_calc, thresholds)
        hubs, links, satellites = matcher.classify_lemmas(lemmas, "uploaded_image")

        progress(0.8, desc="Génération du schéma Data Vault...")

        # ÉTAPE 5: Génération schéma Data Vault
        generator = DataVaultGenerator()
        schema = generator.generate_schema(hubs, links, satellites, "uploaded_image", lemmas)

        # Valider le schéma
        progress(0.9, desc="Validation du schéma...")
        errors = generator.validate_schema(schema)

        # Stocker le schéma pour l'export
        last_schema = schema

        progress(0.95, desc="Formatage des résultats...")

        # Préparer l'affichage des lemmes
        lemmas_str = f"### Lemmes extraits ({len(lemmas)})\n\n"
        lemmas_str += "```\n" + ", ".join(lemmas) + "\n```"

        # Organiser les résultats par Hub
        results_by_hub_str = _format_results_by_hub(hubs, links, satellites)

        # Statistiques
        stats = schema.get_statistics()
        elapsed_time = time.time() - start_time

        # Mapper les valeurs pour l'affichage
        similarity_names = {
            "hybrid": "Hybride (Jaro-Winkler + Embeddings)",
            "lexical": "Lexical (Jaro-Winkler)",
            "semantic": "Sémantique (Embeddings)"
        }

        stats_str = f"""
### Statistiques

**Configuration**
- Modèle: {selected_model}
- Ontologie: {selected_ontology}
- Similarité: {similarity_names.get(selected_similarity, selected_similarity)}
- Temps: {elapsed_time:.2f}s

**Éléments détectés**
- Hubs (entités): {stats["total_hubs"]}
- Links (relations): {stats["total_links"]}
- Satellites (attributs): {stats["total_satellites"]}

**Confiance moyenne**
- Hubs: {stats['average_confidence']['hubs']:.2%}
- Links: {stats['average_confidence']['links']:.2%}
- Satellites: {stats['average_confidence']['satellites']:.2%}

"""

        #if errors:
        #    stats_str += f"\n\n**Avertissements** ({len(errors)})\n" + "\n".join([f"- {e}" for e in errors[:5]])

        progress(1.0, desc="Analyse terminée")

        return "✅ **Analyse terminée avec succès**", lemmas_str, results_by_hub_str, stats_str

    except Exception as e:
        progress(1.0, desc="Erreur lors de l'analyse")
        return f"❌ Erreur lors de l'analyse: {str(e)}", "", "", ""


def process_multiple_images(
    uploaded_files: List,
    selected_model: str,
    selected_ontology: str,
    selected_similarity: str,
    threshold_entities: float,
    threshold_relations: float,
    threshold_attributes: float,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Traite plusieurs images uploadées.

    Args:
        uploaded_files: Liste des fichiers images uploadés
        selected_model: Modèle LLM à utiliser
        selected_ontology: Fichier d'ontologie à utiliser
        selected_similarity: Algorithme de similarité à utiliser
        threshold_entities: Seuil pour entités
        threshold_relations: Seuil pour relations
        threshold_attributes: Seuil pour attributs
        progress: Gestionnaire de progression Gradio

    Returns:
        Tuple (status_msg, results_summary)
    """
    global batch_schemas

    # Initialiser la progression immédiatement
    progress(0, desc="Démarrage du traitement par lot...")

    if not uploaded_files:
        return "❌ Aucun fichier uploadé", ""

    try:
        # Convertir en liste si un seul fichier
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        # Filtrer les fichiers None
        image_files = [f for f in uploaded_files if f is not None]

        if not image_files:
            return "❌ Aucune image valide uploadée", ""

        total_images = len(image_files)
        print(f"\n📁 Traitement de {total_images} images uploadées")

        progress(0.05, desc=f"Préparation pour traiter {total_images} images...")

        # Charger l'ontologie une seule fois
        progress(0.1, desc="Chargement de l'ontologie...")
        available_ontologies = config.get_available_ontologies()
        ontology_path = available_ontologies.get(selected_ontology, config.ONTOLOGY_PATH)
        current_ontology = OntologyLoader(ontology_path)

        # Initialiser le calculateur de similarité une seule fois
        progress(0.15, desc="Initialisation du calculateur de similarité...")
        current_similarity_calc = SimilarityCalculator(
            embedding_model=config.EMBEDDING_MODEL,
            ollama_base_url=config.OLLAMA_BASE_URL,
            algorithm=selected_similarity
        )

        # Initialiser l'extracteur LLM
        progress(0.2, desc="Connexion au modèle LLM...")
        current_extractor = LLMExtractor(config.OLLAMA_BASE_URL, selected_model)

        # Thresholds
        thresholds = {
            "entities": threshold_entities,
            "relations": threshold_relations,
            "attributes": threshold_attributes
        }

        # Réinitialiser les schémas batch
        batch_schemas = {}

        # Variables pour statistiques globales
        total_hubs = 0
        total_links = 0
        total_satellites = 0
        successful_images = 0
        failed_images = 0

        # Dictionnaire pour stocker les lemmes de chaque image
        image_lemmas = {}

        start_time = time.time()

        # Traiter chaque image
        for idx, image_file in enumerate(image_files, 1):
            # Obtenir le chemin du fichier uploadé
            image_path = image_file if isinstance(image_file, str) else image_file.name
            image_name = os.path.basename(image_path)

            # Mise à jour de la progression (de 0.2 à 0.95)
            progress_base = 0.2 + ((idx - 1) / total_images) * 0.75
            progress(progress_base, desc=f"📸 Image {idx}/{total_images}: {image_name}")

            try:
                # Extraction de lemmes
                progress(progress_base + 0.1 * 0.75 / total_images, desc=f"🔍 Extraction lemmes: {image_name}")
                lemmas = current_extractor.extract_lemmas(image_path)

                if not lemmas:
                    print(f"⚠️  Aucun lemme extrait pour {image_name}")
                    failed_images += 1
                    image_lemmas[image_name] = []
                    continue

                # Stocker les lemmes
                image_lemmas[image_name] = lemmas
                print(f"✅ {image_name}: {len(lemmas)} lemmes extraits")

                # Classification ontologique
                progress(progress_base + 0.3 * 0.75 / total_images, desc=f"🏷️  Classification: {image_name}")
                matcher = OntologyMatcher(current_ontology, current_similarity_calc, thresholds)
                hubs, links, satellites = matcher.classify_lemmas(lemmas, image_name)

                # Génération schéma Data Vault
                progress(progress_base + 0.6 * 0.75 / total_images, desc=f"💾 Génération schéma: {image_name}")
                generator = DataVaultGenerator()
                schema = generator.generate_schema(hubs, links, satellites, image_name, lemmas)

                # Stocker le schéma
                batch_schemas[image_name] = schema

                # Mettre à jour les statistiques
                stats = schema.get_statistics()
                total_hubs += stats["total_hubs"]
                total_links += stats["total_links"]
                total_satellites += stats["total_satellites"]
                successful_images += 1

                print(f"✅ {image_name}: {stats['total_hubs']} hubs, {stats['total_links']} links, {stats['total_satellites']} satellites")

            except Exception as e:
                print(f"❌ Erreur lors du traitement de {image_name}: {str(e)}")
                failed_images += 1
                image_lemmas[image_name] = []
                continue

        elapsed_time = time.time() - start_time
        progress(0.95, desc="Génération du résumé des résultats...")

        # (Le résumé sera généré ci-dessous)

        progress(1.0, desc="Traitement terminé")

        # Mapper les valeurs pour l'affichage
        similarity_names = {
            "hybrid": "Hybride (Jaro-Winkler + Embeddings)",
            "lexical": "Lexical (Jaro-Winkler)",
            "semantic": "Sémantique (Embeddings)"
        }

        # Générer le résumé
        results_summary = f"""
## Résultats du traitement par lot

### Configuration
- Nombre d'images: {total_images}
- Modèle: {selected_model}
- Ontologie: {selected_ontology}
- Similarité: {similarity_names.get(selected_similarity, selected_similarity)}

### Statistiques globales
- Images traitées avec succès: {successful_images}/{total_images}
- Images en erreur: {failed_images}/{total_images}
- Temps total: {elapsed_time:.2f}s
- Temps moyen par image: {elapsed_time/total_images:.2f}s

### Éléments détectés (total)
- Hubs (entités): {total_hubs}
- Links (relations): {total_links}
- Satellites (attributs): {total_satellites}

### Détails par image
"""

        # Ajouter les détails de chaque image
        for image_name in image_lemmas.keys():
            lemmas = image_lemmas.get(image_name, [])

            results_summary += f"\n---\n\n### 📸 {image_name}\n\n"

            # Afficher les lemmes
            if lemmas:
                results_summary += f"**Lemmes extraits ({len(lemmas)}):**\n\n"
                results_summary += "```\n" + ", ".join(lemmas) + "\n```\n\n"
            else:
                results_summary += "⚠️ Aucun lemme extrait\n\n"

            # Afficher les statistiques si le schéma existe
            if image_name in batch_schemas:
                schema = batch_schemas[image_name]
                stats = schema.get_statistics()
                results_summary += f"""**Résultats Data Vault:**
- Hubs (entités): {stats['total_hubs']}
- Links (relations): {stats['total_links']}
- Satellites (attributs): {stats['total_satellites']}
- Confiance moyenne: Hubs {stats['average_confidence']['hubs']:.2%}, Links {stats['average_confidence']['links']:.2%}, Satellites {stats['average_confidence']['satellites']:.2%}
"""
            else:
                results_summary += "❌ Traitement échoué\n"

        status_msg = f"✅ **Traitement terminé**\n\n{successful_images} images traitées avec succès sur {total_images}"

        return status_msg, results_summary

    except Exception as e:
        return f"❌ Erreur lors du traitement: {str(e)}", ""


def export_batch_results(export_format: str) -> Tuple[str, Optional[str]]:
    """
    Exporte tous les schémas du traitement par lot.

    Args:
        export_format: Format d'export ("JSON", "RDF", "SQL")

    Returns:
        Tuple (message, chemin_fichier)
    """
    global batch_schemas

    if not batch_schemas:
        return "❌ Aucun schéma à exporter. Traitez d'abord un répertoire d'images.", None

    try:
        export_dir = Path(config.EXPORT_PATH)
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_dir = export_dir / f"batch_{timestamp}"
        batch_dir.mkdir(exist_ok=True)

        exported_count = 0

        for image_name, schema in batch_schemas.items():
            # Nettoyer le nom de fichier
            clean_name = Path(image_name).stem

            if export_format == "JSON":
                exporter = JSONExporter()
                output_path = batch_dir / f"{clean_name}.json"
                exporter.export(schema, str(output_path))

            elif export_format == "RDF/Turtle":
                exporter = RDFExporter()
                output_path = batch_dir / f"{clean_name}.ttl"
                exporter.export(schema, str(output_path), format="turtle")

            elif export_format == "SQL":
                exporter = SQLExporter()
                output_path = batch_dir / f"{clean_name}.sql"
                exporter.export(schema, str(output_path))

            else:
                return "❌ Format d'export invalide", None

            exported_count += 1

        message = f"✅ Export {export_format} réussi\n\n{exported_count} fichiers exportés dans:\n📁 {batch_dir}"

        # Créer un fichier zip pour faciliter le téléchargement
        zip_path = str(batch_dir) + ".zip"
        shutil.make_archive(str(batch_dir), 'zip', batch_dir)

        return message, zip_path

    except Exception as e:
        return f"❌ Erreur lors de l'export: {str(e)}", None


def _format_results_by_hub(hubs: list, links: list, satellites: list) -> str:
    """
    Formate les résultats en structure Data Vault linéaire.

    Args:
        hubs: Liste des Hubs
        links: Liste des Links
        satellites: Liste des Satellites

    Returns:
        String formaté
    """
    if not hubs:
        return "Aucune entité détectée"

    output = "## Structure Data Vault\n\n"
    output += "```\n"

    # Identifier le hub plante et le hub problème
    hub_plante = None
    hub_probleme = None

    for hub in hubs:
        if hub.entity_type == 'plante':
            hub_plante = hub
        elif hub.entity_type in ('maladie', 'ravageur'):
            hub_probleme = hub

    # Afficher le Hub plante
    if hub_plante:
        output += f"HUB plante: {hub_plante.business_key}\n\n"

    # Afficher le Link et le Hub problème
    if links:
        link = links[0]
        output += f"   LINK relation: {link.relation_type}\n\n"

    if hub_probleme:
        output += f"   HUB {hub_probleme.entity_type}: {hub_probleme.business_key}\n\n"

    # Afficher tous les Satellites
    for sat in satellites:
        # Vérifier si c'est un match par similarité (score < 1.0)
        if sat.confidence_score < 1.0:
            output += f"   SATELLITE {sat.attribute_name} (score: {sat.confidence_score:.2f}): {sat.attribute_value}\n"
        else:
            output += f"   SATELLITE {sat.attribute_name}: {sat.attribute_value}\n"

    # Afficher le résumé du Link créé
    if hub_plante and hub_probleme and links:
        link = links[0]
        output += f"\n   LINK créé: {hub_plante.business_key} --{link.relation_type}--> {hub_probleme.business_key}\n"

    output += "```\n"

    return output


def export_schema(export_format: str) -> Tuple[str, Optional[str]]:
    """
    Exporte le dernier schéma généré.

    Args:
        export_format: Format d'export ("JSON", "RDF", "SQL")

    Returns:
        Tuple (message, chemin_fichier)
    """
    global last_schema

    if last_schema is None:
        return "❌ Aucun schéma à exporter. Analysez d'abord une image.", None

    try:
        # Créer le dossier exports
        export_dir = Path(config.EXPORT_PATH)
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if export_format == "JSON":
            exporter = JSONExporter()
            output_path = export_dir / f"schema_{timestamp}.json"
            exporter.export(last_schema, str(output_path))
            message = f"✅ Export JSON réussi"

        elif export_format == "RDF/Turtle":
            exporter = RDFExporter()
            output_path = export_dir / f"schema_{timestamp}.ttl"
            exporter.export(last_schema, str(output_path), format="turtle")
            message = f"✅ Export RDF/Turtle réussi"

        elif export_format == "SQL":
            exporter = SQLExporter()
            output_path = export_dir / f"schema_{timestamp}.sql"
            exporter.export(last_schema, str(output_path))
            message = f"✅ Export SQL réussi"

        else:
            return "❌ Format d'export invalide", None

        return f"{message}\n📁 Fichier: {output_path}", str(output_path)

    except Exception as e:
        return f"❌ Erreur lors de l'export: {str(e)}", None


def get_ontology_info() -> str:
    """Retourne les informations sur l'ontologie."""
    stats = ontology.get_statistics()

    info = f"""
# Ontologie - Plantes Agricoles du Burkina Faso

### Statistiques

- Triplets RDF: {stats['total_triples']}
- Concepts: {stats['concepts']} (Classes: {stats['classes']}, Instances: {stats['individuals']})
- Relations: {stats['relations']}
- Attributs: {stats['attributes']}

### Domaine

**Plantes couvertes:**
Maïs, Oignon, Tomate

**Éléments modélisés:**
- Maladies (fongiques, bactériennes, virales, ravageurs, stress)
- Symptômes (nécrose, chlorose, flétrissement, taches foliaires)
- Caractéristiques morphologiques (couleur, forme, texture, nervation)

### Fichier

`{config.ONTOLOGY_PATH}`
"""
    return info


# ========== CONSTRUCTION DE L'INTERFACE GRADIO ==========

# Thème personnalisé sobre et moderne
theme = gr.themes.Monochrome(
    primary_hue="slate",
    secondary_hue="zinc",
    neutral_hue="stone",
    radius_size="sm",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#fafafa",
    body_background_fill_dark="#0f0f0f",
    block_background_fill="white",
    block_border_width="1px",
    block_label_text_weight="500",
    button_primary_background_fill="#18181b",
    button_primary_background_fill_hover="#27272a",
)

with gr.Blocks(title="Extraction de Connaissances - Plantes", theme=theme, css="""
    .gradio-container {max-width: 1400px !important;}
    h1 {font-size: 1.75rem !important; font-weight: 600 !important; margin-bottom: 0.5rem !important;}
    h2 {font-size: 1.25rem !important; font-weight: 500 !important;}
    h3 {font-size: 1.1rem !important; font-weight: 500 !important; color: #52525b !important;}
    .prose {color: #3f3f46 !important;}
""") as app:
    gr.Markdown("""
    # Extraction de Connaissances
    ### Analyse automatique d'images de plantes agricoles

    Maïs, Oignon, Tomate
    """)

    # ========== ONGLET 1: ANALYSE ==========
    with gr.Tab("Analyse"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Image", height=300)

                with gr.Accordion("Configuration", open=False):
                    model_selector = gr.Dropdown(
                        choices=["llava:7b", "llava:13b", "qwen3-vl:latest", "llama3.2-vision:latest"],
                        value="llava:7b",
                        label="Modèle",
                        info="Sélectionner le modèle"
                    )

                    # Liste des ontologies disponibles
                    available_ontologies = config.get_available_ontologies()
                    ontology_choices = list(available_ontologies.keys())
                    default_ontology = Path(config.ONTOLOGY_PATH).name

                    ontology_selector = gr.Dropdown(
                        choices=ontology_choices,
                        value=default_ontology if default_ontology in ontology_choices else (ontology_choices[0] if ontology_choices else None),
                        label="Ontologie",
                        info="Sélectionner le fichier d'ontologie"
                    )

                    similarity_selector = gr.Dropdown(
                        choices=[
                            ("Hybride (Jaro-Winkler + Embeddings)", "hybrid"),
                            ("Jaro-Winkler + Cosinus", "jaro_cosine"),
                            ("Lexical (Jaro-Winkler)", "lexical"),
                            ("Sémantique (Embeddings)", "semantic"),
                            ("Cosinus (N-grams)", "cosine"),
                            ("Jaro-Winkler", "jaro_winkler")
                        ],
                        value=config.SIMILARITY_ALGORITHM,
                        label="Mesure de similarité",
                        info="Algorithme de comparaison des lemmes"
                    )

                    threshold_e = gr.Slider(
                        0.0, 1.0, config.THRESHOLD_ENTITIES, step=0.05,
                        label="Seuil hub (entités)"
                    )
                    threshold_r = gr.Slider(
                        0.0, 1.0, config.THRESHOLD_RELATIONS, step=0.05,
                        label="Seuil link (relations)"
                    )
                    threshold_a = gr.Slider(
                        0.0, 1.0, config.THRESHOLD_ATTRIBUTES, step=0.05,
                        label="Seuil satellite (attributs)"
                    )

                analyze_btn = gr.Button("Analyser", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_box = gr.Markdown(label="Statut", value="")
                lemmas_output = gr.Markdown()
                stats_output = gr.Markdown()

        with gr.Row():
            results_by_hub = gr.Markdown()

        # Effacer les sorties puis lancer l'analyse
        analyze_btn.click(
            fn=clear_outputs,
            inputs=None,
            outputs=[status_box, lemmas_output, results_by_hub, stats_output]
        ).then(
            fn=analyze_image,
            inputs=[image_input, model_selector, ontology_selector, similarity_selector, threshold_e, threshold_r, threshold_a],
            outputs=[status_box, lemmas_output, results_by_hub, stats_output],
            show_progress="full"
        )

    # ========== ONGLET 2: TRAITEMENT PAR LOT ==========
    with gr.Tab("Traitement par lot"):
        with gr.Row():
            with gr.Column(scale=1):
                multiple_images_input = gr.File(
                    label="Sélectionner plusieurs images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath"
                )

                with gr.Accordion("Configuration", open=False):
                    batch_model_selector = gr.Dropdown(
                        choices=["llava:7b", "llava:13b", "qwen3-vl:latest", "llama3.2-vision:latest"],
                        value="llava:7b",
                        label="Modèle",
                        info="Sélectionner le modèle"
                    )

                    # Liste des ontologies disponibles
                    available_ontologies = config.get_available_ontologies()
                    ontology_choices = list(available_ontologies.keys())
                    default_ontology = Path(config.ONTOLOGY_PATH).name

                    batch_ontology_selector = gr.Dropdown(
                        choices=ontology_choices,
                        value=default_ontology if default_ontology in ontology_choices else (ontology_choices[0] if ontology_choices else None),
                        label="Ontologie",
                        info="Sélectionner le fichier d'ontologie"
                    )

                    batch_similarity_selector = gr.Dropdown(
                        choices=[
                            ("Hybride (Jaro-Winkler + Embeddings)", "hybrid"),
                            ("Jaro-Winkler + Cosinus", "jaro_cosine"),
                            ("Lexical (Jaro-Winkler)", "lexical"),
                            ("Sémantique (Embeddings)", "semantic"),
                            ("Cosinus (N-grams)", "cosine"),
                            ("Jaro-Winkler", "jaro_winkler")
                        ],
                        value=config.SIMILARITY_ALGORITHM,
                        label="Mesure de similarité",
                        info="Algorithme de comparaison des lemmes"
                    )

                    batch_threshold_e = gr.Slider(
                        0.0, 1.0, config.THRESHOLD_ENTITIES, step=0.05,
                        label="Seuil hub (entités)"
                    )
                    batch_threshold_r = gr.Slider(
                        0.0, 1.0, config.THRESHOLD_RELATIONS, step=0.05,
                        label="Seuil link (relations)"
                    )
                    batch_threshold_a = gr.Slider(
                        0.0, 1.0, config.THRESHOLD_ATTRIBUTES, step=0.05,
                        label="Seuil satellite (attributs)"
                    )

                process_btn = gr.Button("Traiter les images", variant="primary", size="lg")

                gr.Markdown("""
                ### Instructions
                1. Cliquez sur "Sélectionner plusieurs images"
                2. Sélectionnez toutes les images à traiter (Ctrl+Clic ou Shift+Clic)
                3. Cliquez sur "Traiter les images"

                ### Formats supportés
                JPG, JPEG, PNG, BMP, GIF, TIFF

                ### Note
                Le traitement peut prendre plusieurs minutes selon le nombre d'images.
                """)

            with gr.Column(scale=2):
                batch_status_box = gr.Markdown(label="Statut", value="")
                batch_results_output = gr.Markdown()

        # Action du bouton de traitement
        process_btn.click(
            fn=process_multiple_images,
            inputs=[
                multiple_images_input,
                batch_model_selector,
                batch_ontology_selector,
                batch_similarity_selector,
                batch_threshold_e,
                batch_threshold_r,
                batch_threshold_a
            ],
            outputs=[batch_status_box, batch_results_output],
            show_progress="full"
        )

        # Section Export des résultats batch
        gr.Markdown("---")
        gr.Markdown("## Export des résultats")

        with gr.Row():
            with gr.Column(scale=1):
                batch_export_format = gr.Radio(
                    choices=["JSON", "RDF/Turtle", "SQL"],
                    value="JSON",
                    label="Format"
                )
                batch_export_btn = gr.Button("Exporter tous les résultats", variant="primary")

            with gr.Column(scale=2):
                batch_export_message = gr.Textbox(label="Statut", lines=3, show_label=False)
                batch_export_file = gr.File(label="Téléchargement (ZIP)")

        batch_export_btn.click(
            fn=export_batch_results,
            inputs=[batch_export_format],
            outputs=[batch_export_message, batch_export_file]
        )

    # ========== ONGLET 3: EXPORT (Image unique) ==========
    with gr.Tab("Export"):
        with gr.Row():
            with gr.Column(scale=1):
                export_format = gr.Radio(
                    choices=["JSON", "RDF/Turtle", "SQL"],
                    value="JSON",
                    label="Format"
                )
                export_btn = gr.Button("Exporter", variant="primary")

            with gr.Column(scale=2):
                export_message = gr.Textbox(label="Statut", lines=2, show_label=False)
                export_file = gr.File(label="Téléchargement")

        export_btn.click(
            fn=export_schema,
            inputs=[export_format],
            outputs=[export_message, export_file]
        )

    # ========== ONGLET 4: ONTOLOGIE ==========
    with gr.Tab("Ontologie"):
        ontology_info = gr.Markdown(value=get_ontology_info())

    # ========== ONGLET 5: INFORMATIONS ==========
    with gr.Tab("Info"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Pipeline

                1. Extraction de lemmes (modèle vision)
                2. Classification ontologique
                3. Calcul de similarité
                4. Génération schéma Data Vault

                ### Technologies

                - **LLM**: Ollama
                - **Ontologie**: RDF/OWL
                - **Similarité**: Jaro-Winkler + embeddings
                - **Interface**: Gradio
                - **Data Vault**: Hubs, Links, Satellites
                """)

            with gr.Column():
                gr.Markdown(f"""
                ### Configuration

                **Modèles disponibles:**
                - llava:7b (4-5 GB)
                - qwen2.5vl:3b (10 GB)
                - qwen2.5vl:7b (12 GB)

                **Paramètres:**
                - URL: `{config.OLLAMA_BASE_URL}`
                - Modèle: `{config.LLAVA_MODEL}`
                - Embeddings: `{config.EMBEDDING_MODEL}`

                **Seuils par défaut:**
                - Entités: {config.THRESHOLD_ENTITIES}
                - Relations: {config.THRESHOLD_RELATIONS}
                - Attributs: {config.THRESHOLD_ATTRIBUTES}

                Version 1.0.0
                """)

# ========== LANCEMENT DE L'APPLICATION ==========

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Lancement de l'interface Gradio")
    print("=" * 60)

    app.queue().launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        theme=theme
    )
