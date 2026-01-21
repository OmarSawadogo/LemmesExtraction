"""
Module d'extraction de lemmes via LLaVA (Ollama).
Analyse les images de plantes et extrait des lemmes descriptifs.
"""

import base64
from pathlib import Path
from typing import List, Optional
import re
from unidecode import unidecode

try:
    import ollama
except ImportError:
    print("⚠️  Module ollama non installé. Installation: pip install ollama")
    ollama = None


class LLMExtractor:
    """
    Extracteur de lemmes utilisant LLaVA via Ollama.
    """

    EXTRACTION_PROMPT = """Tu es un expert phytopathologiste. Analyse cette image de plante agricole et extrais les informations structurees.

=== STRUCTURE DE SORTIE ===
Tu dois extraire exactement 4 categories de lemmes dans cet ordre:

1. PLANTE: Une seule parmi [corn, onion, tomato]
2. RELATION: Une seule parmi [has_disease, has_infestation, has_health_status]
3. PROBLEME: Le diagnostic (maladie OU ravageur OU "saine")
4. ATTRIBUTS: Caracteristiques observees (symptomes, etat, morphologie)

=== VOCABULAIRE AUTORISE ===

PLANTES: corn, onion, tomato

MALADIES:
- corn: helminthosporiose, rouille, fusariose, curvulariose, striure, virose, stress_abiotique
- onion: alternariose, mildiou, pourriture_blanche, fusariose, bacteriose, stress_abiotique
- tomato: alternariose, mildiou, fusariose, bacterial_wilt, virus_tylcv, stress_abiotique

RAVAGEURS:
- corn: foreur_tige, chenille_legionnaire, puceron, cicadelle
- onion: thrips, mouche_oignon, chenille, nematode
- tomato: aleurode, acarien, mineuse, noctuelle, thrips, puceron

SYMPTOMES: chlorose, necrose, fletrissement, tache, lesion, pourriture, deformation, mosaique, galerie, perforation, miellat

ETAT_FOLIAIRE: saine, malade, fletrie, seche, tachetee, chlorotique, necrotique, perforee, enroulee

MORPHOLOGIE:
- couleur: vert_fonce, vert_clair, vert_jaunatre, jaune, brun
- forme: lineaire_lanceolee, tubulaire_cylindrique, composee_imparipennee
- texture: lisse, rugueuse, cireuse, creuse
- nervation: nervation_parallele, nervation_reticulee

=== EXEMPLE COMPLET ===

[IMAGE: Feuilles de mais avec taches brunes concentriques et jaunissement]

ANALYSE:
- Feuilles longues a nervures paralleles = corn
- Taches brunes ovales concentriques = helminthosporiose (maladie fongique)
- Jaunissement autour des lesions = chlorose
- Plante visiblement affectee = malade

SORTIE:
corn, has_disease, helminthosporiose, tache, necrose, chlorose, malade, tachetee, vert_jaunatre, lineaire_lanceolee, lisse, nervation_parallele

=== REGLES ===

1. COHERENCE PLANTE-PROBLEME: Le probleme doit etre compatible avec la plante identifiee
2. COHERENCE PROBLEME-SYMPTOMES: Les symptomes doivent justifier le diagnostic
3. COHERENCE MORPHOLOGIE: La morphologie doit correspondre a la plante (ex: corn = nervation_parallele)
4. SI PLANTE SAINE: Utiliser has_health_status et "saine" comme probleme

=== FORMAT DE SORTIE ===
Reponds UNIQUEMENT avec les lemmes separes par des virgules, sans phrases ni explications.
Ordre: plante, relation, probleme, symptomes, etat_foliaire, morphologie

Lemmes:
"""

    def __init__(self, ollama_url: str, model: str = "llava:13b"):
        """
        Initialise l'extracteur LLM.

        Args:
            ollama_url: URL du serveur Ollama
            model: Nom du modèle à utiliser
        """
        if ollama is None:
            raise ImportError("Le module ollama est requis. Installez-le avec: pip install ollama")

        self.ollama_url = ollama_url
        self.model = model
        self.client = ollama.Client(host=ollama_url)

        print(f"🤖 LLM Extractor initialisé: {model} @ {ollama_url}")

    def check_model_availability(self) -> bool:
        """
        Vérifie que le modèle LLaVA est disponible.

        Returns:
            bool: True si le modèle est disponible
        """
        try:
            response = self.client.list()

            # Le client Ollama retourne un objet ListResponse avec un attribut 'models'
            if hasattr(response, 'models'):
                models_list = response.models
            elif isinstance(response, dict):
                models_list = response.get('models', [])
            else:
                models_list = response

            # Extraire les noms de modèles
            available_models = []
            for model in models_list:
                if hasattr(model, 'model'):
                    # C'est un objet Model avec un attribut 'model'
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict):
                    model_name = model.get('model') or model.get('name') or model.get('id', '')
                    if model_name:
                        available_models.append(model_name)
                elif isinstance(model, str):
                    available_models.append(model)

            # Vérifier si le modèle est disponible (avec ou sans tag)
            model_base = self.model.split(':')[0]
            for available in available_models:
                available_base = available.split(':')[0]
                if self.model == available or model_base == available_base:
                    print(f"✅ Modèle {self.model} disponible")
                    return True

            print(f"⚠️  Modèle {self.model} non trouvé.")
            print(f"   Modèles disponibles: {', '.join(available_models)}")
            print(f"   Téléchargez-le avec: ollama pull {self.model}")
            return False

        except Exception as e:
            print(f"❌ Erreur lors de la vérification du modèle: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_lemmas(self, image_path: str, max_retries: int = 3) -> List[str]:
        """
        Extrait les lemmes descriptifs d'une image de plante.

        Args:
            image_path: Chemin vers l'image
            max_retries: Nombre maximum de tentatives

        Returns:
            Liste de lemmes normalisés
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable: {image_path}")

        # Encoder l'image en base64
        try:
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la lecture de l'image: {e}")

        # Tenter l'extraction avec retry
        for attempt in range(max_retries):
            try:
                print(f"🔍 Analyse de {image_path.name}... (tentative {attempt + 1}/{max_retries})")

                response = self.client.generate(
                    model=self.model,
                    prompt=self.EXTRACTION_PROMPT,
                    images=[image_data],
                    options={
                        'temperature': 0.3,  # Faible pour plus de cohérence
                        'top_p': 0.9,
                    }
                )

                # Extraire le texte de la réponse
                response_text = response.get('response', '')

                # Parser et normaliser les lemmes
                lemmas = self._parse_llava_response(response_text)

                if lemmas:
                    print(f"✅ {len(lemmas)} lemmes extraits de {image_path.name}")
                    return lemmas
                else:
                    print(f"⚠️  Aucun lemme extrait (tentative {attempt + 1})")

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Erreur lors de l'extraction (tentative {attempt + 1}): {e}")

                # Détecter erreur de mémoire insuffisante
                if "system memory" in error_msg.lower() and "available" in error_msg.lower():
                    print(f"\n⚠️  MÉMOIRE INSUFFISANTE!")
                    print(f"   Le modèle {self.model} nécessite plus de RAM que disponible.")
                    print(f"   💡 Solutions:")
                    print(f"      1. Sélectionnez 'llava:7b' dans l'interface (4-5 GB)")
                    print(f"      2. Ou 'llama3.2-vision:latest' (7-8 GB)")
                    print(f"      3. Ou augmentez la RAM Docker dans Docker Desktop\n")
                    raise RuntimeError(
                        f"Mémoire insuffisante pour {self.model}. "
                        f"Veuillez sélectionner un modèle plus petit: llava:7b (4GB) ou llama3.2-vision (7-8GB)"
                    )

                if attempt == max_retries - 1:
                    raise

        return []

    def _parse_llava_response(self, response: str) -> List[str]:
        """
        Parse la réponse de LLaVA et extrait les lemmes.

        Args:
            response: Texte de réponse brut

        Returns:
            Liste de lemmes normalisés
        """
        if not response:
            return []

        # Nettoyer la réponse
        response = response.strip()

        # Extraire la partie après "Lemmes extraits :" si présente
        if ":" in response:
            parts = response.split(":")
            response = parts[-1].strip()

        # Supprimer les explications et garder uniquement les lemmes
        # Les lemmes peuvent être sur plusieurs lignes, on les récupère tous
        lines = response.split('\n')

        # Concaténer toutes les lignes qui contiennent des lemmes (séparés par des virgules)
        lemmas_text = ""
        for line in lines:
            line = line.strip()
            # Ignorer les lignes vides ou les lignes qui semblent être des explications
            if line and ',' in line:
                # Ajouter cette ligne aux lemmes
                lemmas_text += line + ","
            elif line and not any(keyword in line.lower() for keyword in ['exemple', 'règle', 'note', '**', '--']):
                # Ligne sans virgule mais qui pourrait contenir des lemmes
                lemmas_text += line + ","

        # Si aucune ligne avec virgule trouvée, prendre toute la réponse
        if not lemmas_text.strip():
            lemmas_text = response

        # Diviser par virgules
        raw_lemmas = [l.strip() for l in lemmas_text.split(',')]

        # Normaliser chaque lemme
        lemmas = []
        for lemma in raw_lemmas:
            normalized = self._normalize_lemma(lemma)
            if normalized and len(normalized) > 1:  # Ignorer les lemmes trop courts
                lemmas.append(normalized)

        # Dédupliquer en préservant l'ordre
        seen = set()
        unique_lemmas = []
        for lemma in lemmas:
            if lemma not in seen:
                seen.add(lemma)
                unique_lemmas.append(lemma)

        return unique_lemmas

    @staticmethod
    def _normalize_lemma(lemma: str) -> str:
        """
        Normalise un lemme pour le matching.

        Args:
            lemma: Lemme brut

        Returns:
            Lemme normalisé
        """
        # Retirer les caractères spéciaux (sauf underscores)
        lemma = re.sub(r'[^\w\s_-]', '', lemma)

        # Convertir en minuscules
        lemma = lemma.lower()

        # Retirer les accents
        lemma = unidecode(lemma)

        # Remplacer espaces et tirets par underscores
        lemma = lemma.replace(' ', '_')
        lemma = lemma.replace('-', '_')

        # Supprimer les underscores multiples
        lemma = re.sub(r'_+', '_', lemma)

        # Retirer underscores en début/fin
        lemma = lemma.strip('_')

        return lemma

    def batch_extract(self, image_paths: List[str]) -> dict:
        """
        Extrait les lemmes de plusieurs images.

        Args:
            image_paths: Liste de chemins d'images

        Returns:
            Dictionnaire {nom_image: liste_lemmes}
        """
        results = {}

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n📸 Image {i}/{len(image_paths)}")
            try:
                lemmas = self.extract_lemmas(image_path)
                results[Path(image_path).name] = lemmas
            except Exception as e:
                print(f"❌ Erreur pour {image_path}: {e}")
                results[Path(image_path).name] = []

        return results
