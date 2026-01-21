"""
Module d'extraction améliorée de lemmes avec validation stricte et parsing structuré.
Combine l'approche Hub-Link-Satellite avec validation contre l'ontologie.
"""

import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from unidecode import unidecode
from difflib import get_close_matches
from dataclasses import dataclass
from datetime import datetime

try:
    import ollama
except ImportError:
    print("⚠️  Module ollama non installé. Installation: pip install ollama")
    ollama = None


@dataclass
class StructuredPlantAnalysis:
    """Résultat structuré de l'analyse d'une plante."""

    # Identification
    plante: str  # Hub principal
    est_malade: bool

    # Classification Hub-Link-Satellite
    hub: str  # Entité principale (plante)
    link: Optional[str]  # Relation (has_disease, has_infestation, has_health_status)
    satellites: List[str]  # Attributs (maladies, symptômes, caractéristiques)

    # Détails
    type_probleme: Optional[str]  # maladie ou ravageur
    nom_probleme: Optional[str]  # Nom spécifique de la maladie/ravageur
    symptomes: List[str]  # Liste des symptômes observés
    etat_feuille: str  # État général des feuilles
    couleur_feuille: str  # Couleur dominante

    # Métadonnées
    lemmes_bruts: List[str]  # Lemmes originaux extraits
    lemmes_valides: List[str]  # Lemmes après validation
    lemmes_corriges: Dict[str, str]  # {lemme_original: lemme_corrigé}
    confidence_score: float  # Score de confiance global
    timestamp: str

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "identification": {
                "plante": self.plante,
                "est_malade": self.est_malade,
                "type_probleme": self.type_probleme,
                "nom_probleme": self.nom_probleme
            },
            "classification": {
                "hub": self.hub,
                "link": self.link,
                "satellites": self.satellites
            },
            "observations": {
                "symptomes": self.symptomes,
                "etat_feuille": self.etat_feuille,
                "couleur_feuille": self.couleur_feuille
            },
            "lemmes": {
                "bruts": self.lemmes_bruts,
                "valides": self.lemmes_valides,
                "corriges": self.lemmes_corriges
            },
            "metadata": {
                "confidence_score": self.confidence_score,
                "timestamp": self.timestamp
            }
        }


class OntologyValidator:
    """Validateur de lemmes contre l'ontologie des plantes du Burkina Faso."""

    # Définition de l'ontologie (Hub-Link-Satellite)
    HUBS = {
        "plantes": ["corn", "onion", "tomato", "mais", "oignon", "tomate"]
    }

    LINKS = {
        "relations": ["has_disease", "has_infestation", "has_health_status",
                     "a_maladie", "a_infestation", "a_etat_sante"]
    }

    SATELLITES = {
        # États de santé globaux (IMPORTANT: inclure sain/saine en premier)
        "etats_sante": ["sain", "saine", "malade", "stresse", "infeste", "vigoureux"],

        # Maladies spécifiques par plante
        "maladies_corn": ["fusariose", "helminthosporiose", "rouille", "curvulariose",
                         "striure", "virose", "stress_abiotique"],
        "maladies_onion": ["alternariose", "mildiou", "pourriture_blanche", "fusariose",
                          "bacteriose", "stress_abiotique"],
        "maladies_tomato": ["alternariose", "mildiou", "fusariose", "bacterial_wilt",
                           "virus_tylcv", "fletrissement_bacterien", "stress_abiotique"],

        # Ravageurs par plante
        "ravageurs_corn": ["foreur_tige", "chenille_legionnaire", "puceron", "cicadelle"],
        "ravageurs_onion": ["thrips", "mouche", "chenille", "nematode"],
        "ravageurs_tomato": ["aleurode", "acarien", "mineuse", "noctuelle", "thrips",
                            "puceron", "nematode"],

        # Symptômes visuels (présents uniquement si malade)
        "symptomes": ["chlorose", "necrose", "fletrissement", "lesion", "tache", "striure",
                     "mosaique", "pourriture", "galle", "deformation", "nanisme", "toile",
                     "galerie", "miellat", "morsure", "jaunissement", "perforation"],

        # États des feuilles - séparés en sain et malade pour meilleure classification
        "etats_feuille_sain": ["saine", "turgescente", "dressee", "jeune", "mature"],
        "etats_feuille_malade": ["malade", "fletrie", "seche", "cassante", "mourante", "morte",
                                 "chlorotique", "necrotique", "brulee", "decoloree", "rougie",
                                 "enroulee", "tordue", "froissee", "tombante", "aplatie",
                                 "ratatinee", "perforee", "dechiree", "tachetee", "striee",
                                 "marbree", "poudreuse", "collante", "fumagine", "entoilee",
                                 "minee", "senescente"],
        "etats_feuille": ["turgescente", "fletrie", "seche", "cassante", "saine", "malade",
                         "mourante", "morte", "chlorotique", "necrotique", "brulee", "decoloree",
                         "rougie", "enroulee", "tordue", "froissee", "tombante", "dressee",
                         "aplatie", "ratatinee", "perforee", "dechiree", "tachetee", "striee",
                         "marbree", "poudreuse", "collante", "fumagine", "entoilee", "minee",
                         "jeune", "mature", "senescente"],

        # Couleurs des feuilles - séparées en sain et malade
        "couleurs_feuille_sain": ["vert_fonce", "vert_clair", "vert_bleuatre", "vert_moyen"],
        "couleurs_feuille_malade": ["vert_jaunatre", "jaune", "brun", "vert_grisatre"],
        "couleurs_feuille": ["vert_fonce", "vert_clair", "jaune", "brun", "vert_bleuatre",
                            "vert_grisatre", "vert_jaunatre", "vert_moyen"],

        # Formes des feuilles
        "formes_feuille": ["lineaire_lanceolee", "composee_imparipennee", "tubulaire_cylindrique",
                          "simple", "composee", "simple_tubulaire"],

        # Textures
        "textures": ["lisse", "pubescente", "glanduleuse", "lisse_cireuse", "creuse",
                    "legerement_rugueuse"],

        # Nervation
        "nervation": ["nervation_parallele", "nervation_reticulee", "parallele", "reticulee"]
    }

    # Mappings pour normalisation
    PLANT_ALIASES = {
        "mais": "corn",
        "maïs": "corn",
        "oignon": "onion",
        "tomate": "tomato"
    }

    @classmethod
    def get_all_valid_terms(cls) -> List[str]:
        """Retourne tous les termes valides de l'ontologie."""
        all_terms = []
        all_terms.extend(cls.HUBS["plantes"])
        all_terms.extend(cls.LINKS["relations"])
        for category in cls.SATELLITES.values():
            all_terms.extend(category)
        return all_terms

    @classmethod
    def validate_term(cls, value: str, valid_terms: List[str], cutoff: float = 0.6) -> Optional[str]:
        """
        Valide et corrige un terme contre une liste de termes valides.

        Args:
            value: Terme à valider
            valid_terms: Liste des termes valides
            cutoff: Seuil de similarité pour fuzzy matching (0.0-1.0)

        Returns:
            Terme validé/corrigé ou None si invalide
        """
        if not value:
            return None

        # Normaliser
        value = value.lower().replace(" ", "_").replace("-", "_")
        value = unidecode(value)

        # Match exact
        if value in valid_terms:
            return value

        # Match partiel (substring)
        for term in valid_terms:
            if term in value or value in term:
                return term

        # Fuzzy match avec difflib
        matches = get_close_matches(value, valid_terms, n=1, cutoff=cutoff)
        if matches:
            return matches[0]

        return None

    @classmethod
    def identify_plant(cls, lemme: str) -> Optional[str]:
        """Identifie la plante depuis un lemme."""
        # Normaliser d'abord
        lemme_norm = unidecode(lemme.lower())

        # Vérifier les alias
        if lemme_norm in cls.PLANT_ALIASES:
            return cls.PLANT_ALIASES[lemme_norm]

        # Valider contre les plantes
        return cls.validate_term(lemme, cls.HUBS["plantes"])

    @classmethod
    def identify_link(cls, lemme: str) -> Optional[str]:
        """Identifie un LINK depuis un lemme."""
        return cls.validate_term(lemme, cls.LINKS["relations"])

    @classmethod
    def classify_satellite(cls, lemme: str, plant_type: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Classifie un SATELLITE et détermine sa catégorie.

        Args:
            lemme: Lemme à classifier
            plant_type: Type de plante (corn, onion, tomato) pour contexte

        Returns:
            Tuple (lemme_validé, catégorie)
        """
        # Essayer chaque catégorie de satellites
        for category, terms in cls.SATELLITES.items():
            validated = cls.validate_term(lemme, terms)
            if validated:
                return validated, category

        return None, None


class EnhancedLLMExtractor:
    """
    Extracteur de lemmes amélioré avec validation stricte et parsing structuré.
    """

    # Prompt structuré pour extraction avec format clé:valeur
    STRUCTURED_PROMPT = """Analyse cette image de plante agricole du Burkina Faso.

**ÉTAPE 1 - IDENTIFICATION**
Identifier le type de plante: corn (maïs), onion (oignon) ou tomato (tomate)

**ÉTAPE 2 - DIAGNOSTIC SANITAIRE**
Déterminer si la plante est malade: oui ou non

**ÉTAPE 3 - SI MALADE (sinon écrire "aucun")**
- TYPE_PROBLEME: maladie ou ravageur
- NOM_PROBLEME (selon la plante identifiée):
  * corn: fusariose, helminthosporiose, rouille, curvulariose, striure, virose
  * onion: alternariose, mildiou, pourriture_blanche, fusariose, bacteriose
  * tomato: alternariose, mildiou, fusariose, bacterial_wilt, virus_tylcv
- RAVAGEURS (selon la plante):
  * corn: foreur_tige, chenille_legionnaire, puceron, cicadelle
  * onion: thrips, mouche, chenille, nematode
  * tomato: aleurode, acarien, mineuse, noctuelle, thrips, puceron
- SYMPTOMES: chlorose, necrose, fletrissement, tache, striure, mosaique, pourriture, deformation

**ÉTAPE 4 - OBSERVATIONS VISUELLES**
- ETAT_FEUILLE: saine, malade, fletrie, seche, chlorotique, necrotique, tachetee, striee
- COULEUR_FEUILLE: vert_fonce, vert_clair, jaune, brun, vert_jaunatre

**IMPORTANT**: Réponds STRICTEMENT dans ce format avec des underscores:
PLANTE: [corn/onion/tomato]
EST_MALADE: [oui/non]
TYPE_PROBLEME: [maladie/ravageur/aucun]
NOM_PROBLEME: [nom_specifique ou aucun]
SYMPTOME: [symptome_observe ou aucun]
ETAT_FEUILLE: [etat]
COULEUR_FEUILLE: [couleur]

Exemple de réponse valide:
PLANTE: corn
EST_MALADE: oui
TYPE_PROBLEME: maladie
NOM_PROBLEME: helminthosporiose
SYMPTOME: necrose
ETAT_FEUILLE: necrotique
COULEUR_FEUILLE: vert_fonce

Ta réponse:
"""

    # Prompt original (fallback)
    LEMMA_PROMPT = """
Analysez une image de plante agricole provenant du Burkina Faso en vous appuyant sur l'ontologie **planteontolgie**.

Identifiez et extrayez les informations suivantes **exclusivement sous forme de lemmes**, séparés par des virgules, en respectant strictement la hiérarchie Hubs-Links-Satellites :

**HUBS (Classes principales - choisir parmi) :**
* corn, onion, tomato

**LINKS (Relations d'état sanitaire - choisir parmi) :**
* has_disease, has_infestation, has_health_status

**SATELLITES (Propriétés de données - valeurs observables) :**

1. **Maladies spécifiques** (selon la plante identifiée) :
   * corn : fusariose, helminthosporiose, rouille, curvulariose, striure, virose, stress_abiotique
   * onion : alternariose, mildiou, pourriture_blanche, fusariose, bacteriose, stress_abiotique
   * tomato : alternariose, mildiou, fusariose, bacterial_wilt, virus_tylcv, stress_abiotique

2. **Ravageurs spécifiques** (selon la plante identifiée) :
   * corn : foreur_tige, chenille_legionnaire, puceron, cicadelle
   * onion : thrips, mouche, chenille, nematode
   * tomato : aleurode, acarien, mineuse, noctuelle, thrips, puceron, nematode

3. **Symptômes visuels** (observés sur la plante) :
   * chlorose, necrose, flétrissement, lesion, tache, striure, mosaïque, pourriture, galle, deformation, nanisme, toile, galerie, miellat, morsure

4. **États des feuilles** (observés) :
   * turgescente, fletrie, seche, cassante, saine, malade, mourante, morte, chlorotique, necrotique, brulee, decoloree, rougie, enroulee, tordue, froissee, tombante, dressee, aplatie, ratatinee, perforee, dechiree, tachetee, striee, marbree, poudreuse, collante, fumagine, entoilee, minee, jeune, mature, senescente

5. **Caractéristiques morphologiques** :
   * Couleur_feuilles : vert_fonce, vert_clair, jaune, brun, vert_bleuatre, vert_grisatre, vert_jaunatre
   * Forme_feuilles : lineaire_lanceolee, composee_imparipennee, tubulaire_cylindrique, simple, composee, simple_tubulaire
   * Texture_feuilles : lisse, pubescente, glanduleuse, lisse_cireuse, creuse, legerement_rugueuse
   * Nervation : nervation_parallele, nervation_reticulee

**Contraintes strictes :**
1. **Toujours commencer par le HUB** (type de plante)
2. **Inclure les LINKS pertinents** basés sur l'état observé
3. **Suivre avec les SATELLITES** correspondant aux observations
4. Utiliser uniquement des **lemmes compatibles avec l'ontologie**
5. Employer des **underscores** pour les expressions composées
6. Ne produire **aucune phrase complète**
7. Séparer les lemmes uniquement par des virgules

**Exemple de sortie valide :**
corn, has_disease, helminthosporiose, necrose, tachetee, necrotique, vert_fonce, lineaire_lanceolee, nervation_parallele, lisse

**Lemmes extraits :**

"""

    def __init__(self, ollama_url: str, model: str = "llava:13b", use_structured: bool = True):
        """
        Initialise l'extracteur amélioré.

        Args:
            ollama_url: URL du serveur Ollama
            model: Nom du modèle à utiliser
            use_structured: Si True, utilise le prompt structuré (recommandé)
        """
        if ollama is None:
            raise ImportError("Le module ollama est requis. Installez-le avec: pip install ollama")

        self.ollama_url = ollama_url
        self.model = model
        self.use_structured = use_structured
        self.client = ollama.Client(host=ollama_url)
        self.validator = OntologyValidator()

        print(f"🤖 Enhanced LLM Extractor initialisé: {model} @ {ollama_url}")
        print(f"   Mode: {'Structuré' if use_structured else 'Lemmes libres'}")

    def analyze_plant(self, image_path: str, max_retries: int = 3) -> StructuredPlantAnalysis:
        """
        Analyse complète d'une image de plante avec validation.

        Args:
            image_path: Chemin vers l'image
            max_retries: Nombre de tentatives

        Returns:
            Analyse structurée de la plante
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable: {image_path}")

        # Encoder l'image
        try:
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Erreur lecture image: {e}")

        # Choisir le prompt
        prompt = self.STRUCTURED_PROMPT if self.use_structured else self.LEMMA_PROMPT

        # Extraire avec retry
        for attempt in range(max_retries):
            try:
                print(f"🔍 Analyse structurée de {image_path.name}... (tentative {attempt + 1}/{max_retries})")

                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    images=[image_data],

                )

                response_text = response.get('response', '')

                # Parser selon le mode
                if self.use_structured:
                    result = self._parse_structured_response(response_text, image_path.name)
                else:
                    result = self._parse_lemma_response(response_text, image_path.name)

                if result:
                    print(f"✅ Analyse complète: {result.hub} - {result.link} - {len(result.satellites)} satellites")
                    return result
                else:
                    print(f"⚠️  Échec parsing (tentative {attempt + 1})")

            except Exception as e:
                print(f"❌ Erreur analyse (tentative {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError(f"Échec analyse après {max_retries} tentatives")

    def _parse_structured_response(self, response: str, image_name: str) -> Optional[StructuredPlantAnalysis]:
        """
        Parse une réponse structurée (format clé:valeur).

        Args:
            response: Réponse du modèle
            image_name: Nom de l'image source

        Returns:
            Analyse structurée ou None
        """
        # Extraire les valeurs
        data = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().upper()
                value = value.strip().lower().replace(" ", "_").replace("-", "_")
                value = unidecode(value)

                # Nettoyer les valeurs invalides
                if value in ["aucun", "none", "n/a", "non_applicable", ""]:
                    value = None

                data[key] = value

        # Validation et construction
        try:
            # 1. Identifier la plante (HUB)
            plante_raw = data.get("PLANTE")
            plante = self.validator.identify_plant(plante_raw)

            if not plante:
                print(f"⚠️  Plante non reconnue: {plante_raw}")
                return None

            # 2. État sanitaire
            est_malade = data.get("EST_MALADE") == "oui"

            # 3. Construire le LINK
            link = None
            type_probleme = data.get("TYPE_PROBLEME")

            if est_malade:
                if type_probleme == "ravageur":
                    link = "has_infestation"
                else:
                    link = "has_disease"
            else:
                link = "has_health_status"

            # 4. Construire les SATELLITES
            satellites = []
            lemmes_corriges = {}

            # Ajouter l'état de santé global (sain/malade/infeste)
            if est_malade:
                if type_probleme == "ravageur":
                    satellites.append("infeste")
                else:
                    satellites.append("malade")
            else:
                # Plante saine - ajouter les deux formes pour meilleure correspondance
                satellites.append("sain")

            # Nom du problème (seulement si malade)
            nom_probleme = data.get("NOM_PROBLEME")
            if nom_probleme and nom_probleme not in ["aucun", "none", "n/a", ""]:
                validated, category = self.validator.classify_satellite(nom_probleme, plante)
                if validated:
                    satellites.append(validated)
                    if validated != nom_probleme:
                        lemmes_corriges[nom_probleme] = validated
            else:
                nom_probleme = None

            # Symptôme (seulement si malade)
            symptome = data.get("SYMPTOME")
            symptomes = []
            if symptome and symptome not in ["aucun", "none", "n/a", ""]:
                validated = self.validator.validate_term(symptome, self.validator.SATELLITES["symptomes"])
                if validated:
                    satellites.append(validated)
                    symptomes.append(validated)
                    if validated != symptome:
                        lemmes_corriges[symptome] = validated

            # État feuille - utiliser les catégories appropriées selon l'état de santé
            etat = data.get("ETAT_FEUILLE", "saine" if not est_malade else "malade")
            if est_malade:
                etat_validated = self.validator.validate_term(etat, self.validator.SATELLITES["etats_feuille_malade"])
                if not etat_validated:
                    etat_validated = self.validator.validate_term(etat, self.validator.SATELLITES["etats_feuille"])
            else:
                etat_validated = self.validator.validate_term(etat, self.validator.SATELLITES["etats_feuille_sain"])
                if not etat_validated:
                    etat_validated = self.validator.validate_term(etat, self.validator.SATELLITES["etats_feuille"])

            if etat_validated:
                satellites.append(etat_validated)
                if etat_validated != etat:
                    lemmes_corriges[etat] = etat_validated
            else:
                etat_validated = "saine" if not est_malade else "malade"
                satellites.append(etat_validated)

            # Couleur feuille - utiliser les catégories appropriées
            couleur = data.get("COULEUR_FEUILLE", "vert_fonce" if not est_malade else "vert_jaunatre")
            if est_malade:
                couleur_validated = self.validator.validate_term(couleur, self.validator.SATELLITES["couleurs_feuille_malade"])
                if not couleur_validated:
                    couleur_validated = self.validator.validate_term(couleur, self.validator.SATELLITES["couleurs_feuille"])
            else:
                couleur_validated = self.validator.validate_term(couleur, self.validator.SATELLITES["couleurs_feuille_sain"])
                if not couleur_validated:
                    couleur_validated = self.validator.validate_term(couleur, self.validator.SATELLITES["couleurs_feuille"])

            if couleur_validated:
                satellites.append(couleur_validated)
                if couleur_validated != couleur:
                    lemmes_corriges[couleur] = couleur_validated
            else:
                couleur_validated = "vert_fonce" if not est_malade else "vert_jaunatre"
                satellites.append(couleur_validated)

            # Calculer score de confiance
            total_terms = len(data)
            validated_terms = len(satellites) + 1  # +1 pour le hub
            confidence = validated_terms / max(total_terms, 1)

            # Construire les lemmes
            lemmes_bruts = [v for v in data.values() if v and v != "aucun"]
            lemmes_valides = [plante, link] + satellites

            return StructuredPlantAnalysis(
                plante=plante,
                est_malade=est_malade,
                hub=plante,
                link=link,
                satellites=satellites,
                type_probleme=type_probleme if type_probleme != "aucun" else None,
                nom_probleme=nom_probleme if nom_probleme != "aucun" else None,
                symptomes=symptomes,
                etat_feuille=etat_validated,
                couleur_feuille=couleur_validated,
                lemmes_bruts=lemmes_bruts,
                lemmes_valides=lemmes_valides,
                lemmes_corriges=lemmes_corriges,
                confidence_score=confidence,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"❌ Erreur parsing structuré: {e}")
            return None

    def _parse_lemma_response(self, response: str, image_name: str) -> Optional[StructuredPlantAnalysis]:
        """
        Parse une réponse en format lemmes libres et structure le résultat.

        Args:
            response: Réponse du modèle
            image_name: Nom de l'image

        Returns:
            Analyse structurée ou None
        """
        # Extraire les lemmes comme dans l'extracteur original
        lemmes_bruts = self._extract_raw_lemmas(response)

        if not lemmes_bruts:
            return None

        # Classifier et valider chaque lemme
        hub = None
        link = None
        satellites = []
        lemmes_corriges = {}
        lemmes_valides = []

        for lemme in lemmes_bruts:
            # Essayer d'identifier comme plante (HUB)
            if not hub:
                plant = self.validator.identify_plant(lemme)
                if plant:
                    hub = plant
                    lemmes_valides.append(plant)
                    if plant != lemme:
                        lemmes_corriges[lemme] = plant
                    continue

            # Essayer d'identifier comme LINK
            if not link:
                link_term = self.validator.identify_link(lemme)
                if link_term:
                    link = link_term
                    lemmes_valides.append(link_term)
                    if link_term != lemme:
                        lemmes_corriges[lemme] = link_term
                    continue

            # Essayer comme SATELLITE
            validated, category = self.validator.classify_satellite(lemme, hub)
            if validated:
                satellites.append(validated)
                lemmes_valides.append(validated)
                if validated != lemme:
                    lemmes_corriges[lemme] = validated

        # Si pas de HUB trouvé, échec
        if not hub:
            print("⚠️  Aucun HUB (plante) identifié")
            return None

        # Si pas de LINK, inférer intelligemment
        if not link:
            # Chercher des indices de maladie/ravageur
            has_disease_terms = any(
                self.validator.validate_term(
                    sat,
                    self.validator.SATELLITES.get(f"maladies_{hub}", [])
                ) for sat in satellites
            )
            has_pest_terms = any(
                self.validator.validate_term(
                    sat,
                    self.validator.SATELLITES.get(f"ravageurs_{hub}", [])
                ) for sat in satellites
            )

            # Vérifier si des termes "sain/saine" sont présents
            has_healthy_terms = any(
                sat in ["sain", "saine", "turgescente", "dressee", "vigoureux"]
                for sat in satellites
            )

            # Vérifier si des symptômes de maladie sont présents
            has_symptom_terms = any(
                self.validator.validate_term(sat, self.validator.SATELLITES.get("symptomes", []))
                for sat in satellites
            )

            # Vérifier les couleurs qui indiquent une maladie
            has_sick_colors = any(
                sat in self.validator.SATELLITES.get("couleurs_feuille_malade", [])
                for sat in satellites
            )

            # Vérifier les états foliaires malades
            has_sick_leaf_states = any(
                sat in self.validator.SATELLITES.get("etats_feuille_malade", [])
                for sat in satellites
            )

            if has_disease_terms or (has_symptom_terms and not has_healthy_terms):
                link = "has_disease"
            elif has_pest_terms:
                link = "has_infestation"
            elif has_healthy_terms and not has_symptom_terms and not has_sick_colors and not has_sick_leaf_states:
                link = "has_health_status"
            elif has_sick_colors or has_sick_leaf_states:
                link = "has_disease"  # Probablement malade si couleur/état anormal
            else:
                link = "has_health_status"  # Par défaut, considérer sain si pas d'indices

            lemmes_valides.insert(1, link)

        # Extraire les informations spécifiques
        est_malade = link in ["has_disease", "has_infestation"]
        type_probleme = "maladie" if link == "has_disease" else ("ravageur" if link == "has_infestation" else None)

        # Trouver le nom du problème
        nom_probleme = None
        if est_malade:
            disease_terms = self.validator.SATELLITES.get(f"maladies_{hub}", [])
            pest_terms = self.validator.SATELLITES.get(f"ravageurs_{hub}", [])
            for sat in satellites:
                if sat in disease_terms or sat in pest_terms:
                    nom_probleme = sat
                    break

        # Trouver les symptômes
        symptome_terms = self.validator.SATELLITES["symptomes"]
        symptomes = [sat for sat in satellites if sat in symptome_terms]

        # Trouver état et couleur feuille
        etat_terms = self.validator.SATELLITES["etats_feuille"]
        couleur_terms = self.validator.SATELLITES["couleurs_feuille"]

        etat_feuille = next((sat for sat in satellites if sat in etat_terms), "saine")
        couleur_feuille = next((sat for sat in satellites if sat in couleur_terms), "vert_clair")

        # Score de confiance
        confidence = len(lemmes_valides) / max(len(lemmes_bruts), 1)

        return StructuredPlantAnalysis(
            plante=hub,
            est_malade=est_malade,
            hub=hub,
            link=link,
            satellites=satellites,
            type_probleme=type_probleme,
            nom_probleme=nom_probleme,
            symptomes=symptomes,
            etat_feuille=etat_feuille,
            couleur_feuille=couleur_feuille,
            lemmes_bruts=lemmes_bruts,
            lemmes_valides=lemmes_valides,
            lemmes_corriges=lemmes_corriges,
            confidence_score=confidence,
            timestamp=datetime.now().isoformat()
        )

    def _extract_raw_lemmas(self, response: str) -> List[str]:
        """Extrait les lemmes bruts de la réponse (identique à l'original)."""
        if not response:
            return []

        response = response.strip()

        if ":" in response:
            parts = response.split(":")
            response = parts[-1].strip()

        lines = response.split('\n')
        lemmas_text = ""

        for line in lines:
            line = line.strip()
            if line and ',' in line:
                lemmas_text += line + ","
            elif line and not any(keyword in line.lower() for keyword in ['exemple', 'règle', 'note', '**', '--']):
                lemmas_text += line + ","

        if not lemmas_text.strip():
            lemmas_text = response

        raw_lemmas = [l.strip() for l in lemmas_text.split(',')]

        # Normaliser
        lemmas = []
        for lemma in raw_lemmas:
            normalized = self._normalize_lemma(lemma)
            if normalized and len(normalized) > 1:
                lemmas.append(normalized)

        # Dédupliquer
        seen = set()
        unique_lemmas = []
        for lemma in lemmas:
            if lemma not in seen:
                seen.add(lemma)
                unique_lemmas.append(lemma)

        return unique_lemmas

    @staticmethod
    def _normalize_lemma(lemma: str) -> str:
        """Normalise un lemme (identique à l'original)."""
        lemma = re.sub(r'[^\w\s_-]', '', lemma)
        lemma = lemma.lower()
        lemma = unidecode(lemma)
        lemma = lemma.replace(' ', '_').replace('-', '_')
        lemma = re.sub(r'_+', '_', lemma)
        lemma = lemma.strip('_')
        return lemma

    def extract_lemmas_validated(self, image_path: str, max_retries: int = 3) -> Tuple[List[str], Dict[str, str]]:
        """
        Extrait et valide les lemmes (compatible avec l'API originale).

        Args:
            image_path: Chemin vers l'image
            max_retries: Nombre de tentatives

        Returns:
            Tuple (lemmes_valides, corrections_appliquées)
        """
        analysis = self.analyze_plant(image_path, max_retries)
        return analysis.lemmes_valides, analysis.lemmes_corriges

    def batch_analyze(self, image_paths: List[str]) -> Dict[str, StructuredPlantAnalysis]:
        """
        Analyse un lot d'images.

        Args:
            image_paths: Liste de chemins d'images

        Returns:
            Dictionnaire {nom_image: analyse}
        """
        results = {}

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n📸 Image {i}/{len(image_paths)}")
            try:
                analysis = self.analyze_plant(image_path)
                results[Path(image_path).name] = analysis
            except Exception as e:
                print(f"❌ Erreur pour {image_path}: {e}")
                results[Path(image_path).name] = None

        return results
