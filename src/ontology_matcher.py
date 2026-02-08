"""
Module de classification ontologique guidée.
Implémente les 3 règles de classification : Hubs, Links, Satellites.

Ontologie O = (C, R, A) où:
- C = ensemble des concepts avec labels
- R = ensemble des relations avec labels
- A = ensemble des attributs avec labels
"""

from typing import List, Tuple, Dict, Optional, Set
from src.ontology_loader import OntologyLoader, OntologyConcept, OntologyRelation, OntologyAttribute
from src.similarity_calculator import SimilarityCalculator
from src.models.hub import Hub
from src.models.link import Link
from src.models.satellite import Satellite


class OntologyMatcher:
    """
    Classificateur de lemmes guidé par l'ontologie O = (C, R, A).
    Applique les 3 règles formelles pour identifier H (Hubs), LK (Links), S (Satellites).

    Classification simplifiée et cohérente:
    - HUB: Entités principales (plante + maladie/ravageur identifié)
    - LINK: Relations entre entités (has_disease, has_infestation, has_health_status)
    - SATELLITE: Attributs descriptifs (symptomes, etat_foliaire, morphologie)
    """

    # Vocabulaire contrôlé pour classification directe
    PLANTES = {'corn', 'onion', 'tomato', 'mais', 'oignon', 'tomate'}

    MALADIES = {
        'helminthosporiose', 'rouille', 'fusariose', 'curvulariose', 'striure', 'virose',
        'alternariose', 'mildiou', 'pourriture_blanche', 'bacteriose', 'bacterial_wilt',
        'virus_tylcv', 'stress_abiotique'
    }

    RAVAGEURS = {
        'foreur_tige', 'chenille_legionnaire', 'puceron', 'cicadelle',
        'thrips', 'mouche_oignon', 'chenille', 'nematode',
        'aleurode', 'acarien', 'mineuse', 'noctuelle'
    }

    RELATIONS = {'has_disease', 'has_infestation', 'has_health_status'}

    SYMPTOMES = {
        'chlorose', 'necrose', 'fletrissement', 'tache', 'lesion', 'pourriture',
        'deformation', 'mosaique', 'galerie', 'perforation', 'miellat', 'striure',
        'galle', 'nanisme', 'toile', 'morsure'
    }

    ETAT_FOLIAIRE = {
        'saine', 'malade', 'fletrie', 'seche', 'tachetee', 'chlorotique', 'necrotique',
        'perforee', 'enroulee', 'turgescente', 'cassante', 'mourante', 'morte',
        'brulee', 'decoloree', 'rougie', 'tordue', 'froissee', 'tombante', 'dressee',
        'aplatie', 'ratatinee', 'dechiree', 'striee', 'marbree', 'poudreuse',
        'collante', 'fumagine', 'entoilee', 'minee', 'jeune', 'mature', 'senescente'
    }

    MORPHOLOGIE_COULEUR = {
        'vert_fonce', 'vert_clair', 'vert_jaunatre', 'jaune', 'brun', 'vert_bleuatre',
        'vert_grisatre', 'vert', 'rouge', 'orange', 'noir', 'blanc'
    }

    MORPHOLOGIE_FORME = {
        'lineaire_lanceolee', 'tubulaire_cylindrique', 'composee_imparipennee',
        'simple', 'composee', 'simple_tubulaire', 'ovale', 'lanceolee'
    }

    MORPHOLOGIE_TEXTURE = {
        'lisse', 'rugueuse', 'cireuse', 'creuse', 'pubescente', 'glanduleuse',
        'lisse_cireuse', 'legerement_rugueuse'
    }

    MORPHOLOGIE_NERVATION = {
        'nervation_parallele', 'nervation_reticulee', 'parallele', 'reticulee'
    }

    def __init__(
        self,
        ontology: OntologyLoader,
        similarity_calc: SimilarityCalculator,
        thresholds: Dict[str, float]
    ):
        """
        Initialise le matcher ontologique.

        Args:
            ontology: Ontologie O = (C, R, A)
            similarity_calc: Fonction sim: Σ* × Σ* → [0, 1]
            thresholds: Seuils {θc, θr, θa}
        """
        self.ontology = ontology
        self.similarity_calc = similarity_calc
        self.thresholds = thresholds

        # O = (C, R, A)
        self.C = ontology.get_all_concepts()    # C: concepts
        self.R = ontology.get_all_relations()   # R: relations
        self.A = ontology.get_all_attributes()  # A: attributs

        # Construire les ensembles combinés
        self._build_vocabulary_sets()

        print(f"[ONTO] Ontologie O = (C={len(self.C)}, R={len(self.R)}, A={len(self.A)})")
        print(f"   Seuils: θc={thresholds.get('entities', 0.75)}, θr={thresholds.get('relations', 0.70)}, θa={thresholds.get('attributes', 0.65)}")
        print(f"   Vocabulaire: Hubs={len(self.hub_vocabulary)}, Links={len(self.link_vocabulary)}, Satellites={len(self.satellite_vocabulary)}")

    def _build_vocabulary_sets(self):
        """
        Construit les ensembles de vocabulaire pour la classification.
        Utilise les constantes de classe définies + enrichissement depuis l'ontologie.
        """
        # Vocabulaire Hub = Plantes + Maladies + Ravageurs
        self.hub_vocabulary: Set[str] = set()
        self.hub_vocabulary.update(self.PLANTES)
        self.hub_vocabulary.update(self.MALADIES)
        self.hub_vocabulary.update(self.RAVAGEURS)

        # Vocabulaire Link = Relations
        self.link_vocabulary: Set[str] = set()
        self.link_vocabulary.update(self.RELATIONS)

        # Vocabulaire Satellite = Symptomes + Etat + Morphologie
        self.satellite_vocabulary: Set[str] = set()
        self.satellite_vocabulary.update(self.SYMPTOMES)
        self.satellite_vocabulary.update(self.ETAT_FOLIAIRE)
        self.satellite_vocabulary.update(self.MORPHOLOGIE_COULEUR)
        self.satellite_vocabulary.update(self.MORPHOLOGIE_FORME)
        self.satellite_vocabulary.update(self.MORPHOLOGIE_TEXTURE)
        self.satellite_vocabulary.update(self.MORPHOLOGIE_NERVATION)

        # Enrichir depuis l'ontologie (optionnel)
        self._enrich_from_ontology()

        # Mapping pour type d'attribut satellite
        self.satellite_type_map: Dict[str, str] = {}
        for lemma in self.SYMPTOMES:
            self.satellite_type_map[lemma] = 'symptome'
        for lemma in self.ETAT_FOLIAIRE:
            self.satellite_type_map[lemma] = 'etat_foliaire'
        for lemma in self.MORPHOLOGIE_COULEUR:
            self.satellite_type_map[lemma] = 'couleur'
        for lemma in self.MORPHOLOGIE_FORME:
            self.satellite_type_map[lemma] = 'forme'
        for lemma in self.MORPHOLOGIE_TEXTURE:
            self.satellite_type_map[lemma] = 'texture'
        for lemma in self.MORPHOLOGIE_NERVATION:
            self.satellite_type_map[lemma] = 'nervation'

    def _enrich_from_ontology(self):
        """Enrichit le vocabulaire depuis l'ontologie."""
        # Ajouter les relations de l'ontologie
        for relation in self.R.values():
            lemma = relation.lemma.lower() if relation.lemma else relation.label_normalized
            if lemma and 'has_' in lemma:
                self.link_vocabulary.add(lemma)

    def _classify_lemma(self, lemma: str) -> Tuple[str, Optional[str]]:
        """
        Classifie un lemme en Hub, Link ou Satellite.

        Args:
            lemma: Lemme à classifier

        Returns:
            Tuple (categorie, sous_type) où catégorie est 'hub', 'link', 'satellite' ou 'unknown'
        """
        lemma_lower = lemma.lower().strip()

        # Vérifier si c'est une plante
        if lemma_lower in self.PLANTES:
            return ('hub', 'plante')

        # Vérifier si c'est une maladie
        if lemma_lower in self.MALADIES:
            return ('hub', 'maladie')

        # Vérifier si c'est un ravageur
        if lemma_lower in self.RAVAGEURS:
            return ('hub', 'ravageur')

        # Vérifier si c'est une relation
        if lemma_lower in self.RELATIONS or lemma_lower in self.link_vocabulary:
            return ('link', 'relation')

        # Vérifier si c'est un symptome
        if lemma_lower in self.SYMPTOMES:
            return ('satellite', 'symptome')

        # Vérifier si c'est un état foliaire
        if lemma_lower in self.ETAT_FOLIAIRE:
            return ('satellite', 'etat_foliaire')

        # Vérifier si c'est une couleur
        if lemma_lower in self.MORPHOLOGIE_COULEUR:
            return ('satellite', 'couleur')

        # Vérifier si c'est une forme
        if lemma_lower in self.MORPHOLOGIE_FORME:
            return ('satellite', 'forme')

        # Vérifier si c'est une texture
        if lemma_lower in self.MORPHOLOGIE_TEXTURE:
            return ('satellite', 'texture')

        # Vérifier si c'est une nervation
        if lemma_lower in self.MORPHOLOGIE_NERVATION:
            return ('satellite', 'nervation')

        # Lemme non reconnu
        return ('unknown', None)

    def _find_best_match(self, lemma: str, vocabulary: Set[str], threshold: float) -> Tuple[Optional[str], float]:
        """
        Trouve le meilleur match dans un vocabulaire par similarité.
        Utilise le calculateur de similarité optimisé avec batch processing.

        Args:
            lemma: Lemme à matcher
            vocabulary: Ensemble de vocabulaire
            threshold: Seuil de similarité

        Returns:
            Tuple (best_match, score) ou (None, 0.0)
        """
        if not vocabulary:
            return None, 0.0

        # Utiliser la méthode optimisée du calculateur
        return self.similarity_calc.find_best_match(
            lemma.lower(),
            list(vocabulary),
            threshold
        )

    def classify_lemmas(
        self,
        lemmas: List[str],
        image_source: str
    ) -> Tuple[List[Hub], List[Link], List[Satellite]]:
        """
        Classifie les lemmes en Hubs, Links et Satellites.

        Classification simplifiée:
        - HUB: plante identifiée + maladie/ravageur diagnostiqué
        - LINK: relation entre la plante et le problème
        - SATELLITE: tous les attributs descriptifs (symptomes, etat, morphologie)

        Args:
            lemmas: Liste des lemmes extraits par le LLM
            image_source: Source de l'image

        Returns:
            Tuple (hubs_list, links_list, satellites_list)
        """
        print(f"\n[CLASSIF] Classification de {len(lemmas)} lemmes...")

        # Structures de résultat
        hubs_list: List[Hub] = []
        links_list: List[Link] = []
        satellites_list: List[Satellite] = []

        # Classification par catégorie
        plante_hub = None
        probleme_hub = None
        relation_lemma = None

        # Phase 1: Classification directe de chaque lemme
        for lemma in lemmas:
            category, sub_type = self._classify_lemma(lemma)

            if category == 'hub':
                if sub_type == 'plante':
                    # Créer le Hub plante (principal)
                    plante_hub = Hub(
                        business_key=lemma.lower(),
                        entity_type='plante',
                        record_source=image_source,
                        ontology_uri=f"http://example.org/ontology#{lemma}",
                        confidence_score=1.0
                    )
                    hubs_list.append(plante_hub)
                    print(f"   HUB plante: {lemma}")

                elif sub_type in ('maladie', 'ravageur'):
                    # Créer le Hub problème
                    probleme_hub = Hub(
                        business_key=lemma.lower(),
                        entity_type=sub_type,
                        record_source=image_source,
                        ontology_uri=f"http://example.org/ontology#{lemma}",
                        confidence_score=1.0
                    )
                    hubs_list.append(probleme_hub)
                    print(f"   HUB {sub_type}: {lemma}")

            elif category == 'link':
                relation_lemma = lemma.lower()
                print(f"   LINK relation: {lemma}")

            elif category == 'satellite':
                # On stocke temporairement, on créera les satellites après avoir les hubs
                satellites_list.append((lemma, sub_type))
                print(f"   SATELLITE {sub_type}: {lemma}")

            else:
                # Lemme non reconnu - tenter match par similarité
                self._classify_unknown_lemma(lemma, hubs_list, satellites_list, image_source)

        # Phase 2: Créer les Links entre Hubs
        if plante_hub and probleme_hub:
            # Déterminer le type de relation
            if relation_lemma:
                relation_type = relation_lemma
            elif probleme_hub.entity_type == 'maladie':
                relation_type = 'has_disease'
            elif probleme_hub.entity_type == 'ravageur':
                relation_type = 'has_infestation'
            else:
                relation_type = 'has_health_status'

            link = Link(
                hub_source_key=plante_hub.hub_key,
                hub_target_key=probleme_hub.hub_key,
                relation_type=relation_type,
                record_source=image_source,
                confidence_score=1.0
            )
            links_list.append(link)
            print(f"   LINK créé: {plante_hub.business_key} --{relation_type}--> {probleme_hub.business_key}")

        elif plante_hub and relation_lemma == 'has_health_status':
            # Plante saine, pas de problème
            print(f"   Plante saine détectée")

        # Phase 3: Créer les objets Satellite - distribuer entre hubs
        # Symptomes → hub maladie/ravageur (décrivent le problème)
        # Morphologie + état foliaire → hub plante (décrivent la plante)
        final_satellites: List[Satellite] = []
        DISEASE_SAT_TYPES = {'symptome'}
        PLANT_SAT_TYPES = {'etat_foliaire', 'couleur', 'forme', 'texture', 'nervation', 'description'}

        for sat_data in satellites_list:
            if isinstance(sat_data, tuple):
                lemma, attr_type = sat_data

                # Déterminer le hub parent selon le type d'attribut
                if attr_type in DISEASE_SAT_TYPES and probleme_hub:
                    parent_hub = probleme_hub
                elif plante_hub:
                    parent_hub = plante_hub
                elif hubs_list:
                    parent_hub = hubs_list[0]
                else:
                    continue

                satellite = Satellite(
                    hub_key=parent_hub.hub_key,
                    attribute_name=attr_type,
                    attribute_value=lemma.lower(),
                    record_source=image_source,
                    confidence_score=0.95
                )
                final_satellites.append(satellite)

        print(f"   [RESULTAT] Hubs={len(hubs_list)}, Links={len(links_list)}, Satellites={len(final_satellites)}")

        return hubs_list, links_list, final_satellites

    def _classify_unknown_lemma(
        self,
        lemma: str,
        hubs_list: List[Hub],
        satellites_list: List,
        image_source: str
    ):
        """
        Tente de classifier un lemme non reconnu par similarité.
        """
        theta_c = self.thresholds.get("entities", 0.75)
        theta_a = self.thresholds.get("attributes", 0.65)

        # Essayer de matcher avec le vocabulaire Hub
        best_hub, hub_score = self._find_best_match(lemma, self.hub_vocabulary, theta_c)
        if best_hub:
            # Déterminer le type
            if best_hub in self.PLANTES:
                entity_type = 'plante'
            elif best_hub in self.MALADIES:
                entity_type = 'maladie'
            else:
                entity_type = 'ravageur'

            hub = Hub(
                business_key=lemma.lower(),
                entity_type=entity_type,
                record_source=image_source,
                ontology_uri=f"http://example.org/ontology#{best_hub}",
                confidence_score=hub_score
            )
            hubs_list.append(hub)
            print(f"   HUB (similarité {hub_score:.2f}): {lemma} -> {best_hub}")
            return

        # Essayer de matcher avec le vocabulaire Satellite
        best_sat, sat_score = self._find_best_match(lemma, self.satellite_vocabulary, theta_a)
        if best_sat:
            attr_type = self.satellite_type_map.get(best_sat, 'attribut')
            satellites_list.append((lemma, attr_type))
            print(f"   SATELLITE (similarité {sat_score:.2f}): {lemma} -> {best_sat}")
            return

        # Sinon, classifier comme satellite générique si assez long
        if len(lemma) > 2:
            satellites_list.append((lemma, 'description'))
            print(f"   SATELLITE (générique): {lemma}")

