"""
Module de chargement et parsing de l'ontologie RDF.
Extrait les concepts, relations et attributs depuis le fichier TTL.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from unidecode import unidecode


@dataclass
class OntologyConcept:
    """Représente un concept (classe ou instance) de l'ontologie."""
    uri: str
    label: str
    label_normalized: str
    comment: str
    concept_type: str  # "class" ou "individual"
    parent_classes: List[str]
    # Annotations Data Vault
    is_hub: bool = False
    is_satellite: bool = False
    lemma: str = ""  # Lemme standardisé pour l'extraction


@dataclass
class OntologyRelation:
    """Représente une relation (ObjectProperty) de l'ontologie."""
    uri: str
    label: str
    label_normalized: str
    domain: str
    range: str
    comment: str
    # Annotations Data Vault
    is_link: bool = False
    lemma: str = ""


@dataclass
class OntologyAttribute:
    """Représente un attribut (DatatypeProperty) de l'ontologie."""
    uri: str
    label: str
    label_normalized: str
    domain: str
    datatype: str
    comment: str
    # Annotations Data Vault
    is_satellite: bool = False
    lemma: str = ""


class OntologyLoader:
    """
    Chargeur d'ontologie RDF/OWL.
    Parse le fichier TTL et extrait les concepts, relations et attributs.
    """

    def __init__(self, ontology_path: str):
        """
        Initialise le chargeur d'ontologie.

        Args:
            ontology_path: Chemin vers le fichier TTL
        """
        self.ontology_path = Path(ontology_path)
        self.graph = Graph()
        self.namespace = None

        # Dictionnaires de cache
        self._concepts: Optional[Dict[str, OntologyConcept]] = None
        self._relations: Optional[Dict[str, OntologyRelation]] = None
        self._attributes: Optional[Dict[str, OntologyAttribute]] = None

        # Charger l'ontologie
        self._load_ontology()

    def _load_ontology(self):
        """Charge le fichier TTL dans le graphe RDF."""
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Fichier ontologie introuvable: {self.ontology_path}")

        try:
            self.graph.parse(self.ontology_path, format="turtle")
            print(f"[OK] Ontologie chargee: {len(self.graph)} triplets")

            # Déterminer le namespace principal (supporte # et /)
            for subj, pred, obj in self.graph:
                if isinstance(subj, rdflib.URIRef):
                    uri_str = str(subj)
                    if "#" in uri_str:
                        base_uri = uri_str.split("#")[0] + "#"
                        self.namespace = Namespace(base_uri)
                        break
                    elif "/planteontolgie/" in uri_str:
                        # Format: http://.../planteontolgie/Concept
                        base_uri = uri_str.rsplit("/", 1)[0] + "/"
                        self.namespace = Namespace(base_uri)
                        break

        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de l'ontologie: {e}")

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Normalise un label pour le matching.

        Args:
            label: Label à normaliser

        Returns:
            Label normalisé (lowercase, sans accents, underscores → espaces)
        """
        normalized = label.lower()
        normalized = unidecode(normalized)
        normalized = normalized.replace("_", " ")
        normalized = normalized.strip()
        return normalized

    def get_all_concepts(self) -> Dict[str, OntologyConcept]:
        """
        Extrait tous les concepts (classes et instances) de l'ontologie.

        Returns:
            Dictionnaire {uri: OntologyConcept}
        """
        if self._concepts is not None:
            return self._concepts

        concepts = {}

        # Extraire les classes OWL
        for class_uri in self.graph.subjects(RDF.type, OWL.Class):
            concept = self._extract_concept(class_uri, "class")
            if concept:
                concepts[str(class_uri)] = concept

        # Extraire les instances (individuals)
        for individual_uri in self.graph.subjects(RDF.type, OWL.NamedIndividual):
            concept = self._extract_concept(individual_uri, "individual")
            if concept:
                concepts[str(individual_uri)] = concept

        self._concepts = concepts
        print(f"[C] Concepts charges: {len(concepts)}")
        return concepts

    def _extract_concept(self, uri: rdflib.URIRef, concept_type: str) -> Optional[OntologyConcept]:
        """
        Extrait les informations d'un concept.

        Args:
            uri: URI du concept
            concept_type: "class" ou "individual"

        Returns:
            OntologyConcept ou None
        """
        # Récupérer le label
        label = self._get_label(uri)
        if not label:
            return None

        # Récupérer le commentaire
        comment = ""
        for o in self.graph.objects(uri, RDFS.comment):
            comment = str(o)
            break

        # Récupérer les classes parentes (subClassOf pour classes, rdf:type pour individuals)
        parent_classes = []
        # Pour les classes: rdfs:subClassOf
        for parent in self.graph.objects(uri, RDFS.subClassOf):
            parent_classes.append(str(parent))
        # Pour les individuals: rdf:type (exclure owl:NamedIndividual et owl:Class)
        if concept_type == "individual":
            for rdf_type in self.graph.objects(uri, RDF.type):
                type_str = str(rdf_type)
                if 'NamedIndividual' not in type_str and 'Class' not in type_str:
                    parent_classes.append(type_str)

        # Extraire les annotations Data Vault
        is_hub = self._get_annotation_bool(uri, "isHub")
        is_satellite = self._get_annotation_bool(uri, "isSatellite")
        lemma = self._get_annotation_string(uri, "lemma")

        # Si pas de lemma défini, utiliser le label normalisé
        if not lemma:
            lemma = self._normalize_label(label)

        return OntologyConcept(
            uri=str(uri),
            label=label,
            label_normalized=self._normalize_label(label),
            comment=comment,
            concept_type=concept_type,
            parent_classes=parent_classes,
            is_hub=is_hub,
            is_satellite=is_satellite,
            lemma=lemma
        )

    def get_all_relations(self) -> Dict[str, OntologyRelation]:
        """
        Extrait toutes les relations (ObjectProperties) de l'ontologie.

        Returns:
            Dictionnaire {uri: OntologyRelation}
        """
        if self._relations is not None:
            return self._relations

        relations = {}

        for prop_uri in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            relation = self._extract_relation(prop_uri)
            if relation:
                relations[str(prop_uri)] = relation

        self._relations = relations
        print(f"[R] Relations chargees: {len(relations)}")
        return relations

    def _extract_relation(self, uri: rdflib.URIRef) -> Optional[OntologyRelation]:
        """
        Extrait les informations d'une relation.

        Args:
            uri: URI de la relation

        Returns:
            OntologyRelation ou None
        """
        # Récupérer le label
        label = self._get_label(uri)
        if not label:
            return None

        # Récupérer domain et range
        domain = ""
        for d in self.graph.objects(uri, RDFS.domain):
            domain = str(d)
            break

        range_val = ""
        for r in self.graph.objects(uri, RDFS.range):
            range_val = str(r)
            break

        # Récupérer le commentaire
        comment = ""
        for o in self.graph.objects(uri, RDFS.comment):
            comment = str(o)
            break

        # Extraire les annotations Data Vault
        is_link = self._get_annotation_bool(uri, "isLink")
        lemma = self._get_annotation_string(uri, "lemma")
        if not lemma:
            lemma = self._normalize_label(label)

        return OntologyRelation(
            uri=str(uri),
            label=label,
            label_normalized=self._normalize_label(label),
            domain=domain,
            range=range_val,
            comment=comment,
            is_link=is_link,
            lemma=lemma
        )

    def get_all_attributes(self) -> Dict[str, OntologyAttribute]:
        """
        Extrait tous les attributs (DatatypeProperties) de l'ontologie.

        Returns:
            Dictionnaire {uri: OntologyAttribute}
        """
        if self._attributes is not None:
            return self._attributes

        attributes = {}

        for prop_uri in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            attribute = self._extract_attribute(prop_uri)
            if attribute:
                attributes[str(prop_uri)] = attribute

        self._attributes = attributes
        print(f"[A] Attributs charges: {len(attributes)}")
        return attributes

    def _extract_attribute(self, uri: rdflib.URIRef) -> Optional[OntologyAttribute]:
        """
        Extrait les informations d'un attribut.

        Args:
            uri: URI de l'attribut

        Returns:
            OntologyAttribute ou None
        """
        # Récupérer le label
        label = self._get_label(uri)
        if not label:
            return None

        # Récupérer domain
        domain = ""
        for d in self.graph.objects(uri, RDFS.domain):
            domain = str(d)
            break

        # Récupérer datatype/range
        datatype = ""
        for r in self.graph.objects(uri, RDFS.range):
            datatype = str(r)
            break

        # Récupérer le commentaire
        comment = ""
        for o in self.graph.objects(uri, RDFS.comment):
            comment = str(o)
            break

        # Extraire les annotations Data Vault
        is_satellite = self._get_annotation_bool(uri, "isSatellite")
        lemma = self._get_annotation_string(uri, "lemma")
        if not lemma:
            lemma = self._normalize_label(label)

        return OntologyAttribute(
            uri=str(uri),
            label=label,
            label_normalized=self._normalize_label(label),
            domain=domain,
            datatype=datatype,
            comment=comment,
            is_satellite=is_satellite,
            lemma=lemma
        )

    def _get_label(self, uri: rdflib.URIRef) -> str:
        """
        Récupère le label d'une ressource RDF.

        Args:
            uri: URI de la ressource

        Returns:
            Label (ou fragment URI si pas de label)
        """
        # Essayer de récupérer rdfs:label
        for label in self.graph.objects(uri, RDFS.label):
            return str(label).split("@")[0]  # Retirer la langue si présente

        # Sinon, utiliser le fragment de l'URI
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        elif "/" in uri_str:
            return uri_str.split("/")[-1]

        return uri_str

    def _get_annotation_bool(self, uri: rdflib.URIRef, annotation_name: str) -> bool:
        """
        Récupère une annotation booléenne d'une ressource RDF.

        Args:
            uri: URI de la ressource
            annotation_name: Nom de l'annotation (sans préfixe)

        Returns:
            True si l'annotation existe et vaut "true", False sinon
        """
        if self.namespace:
            annotation_uri = self.namespace[annotation_name]
            for value in self.graph.objects(uri, annotation_uri):
                val_str = str(value).lower()
                return val_str == "true" or val_str == "1"
        return False

    def _get_annotation_string(self, uri: rdflib.URIRef, annotation_name: str) -> str:
        """
        Récupère une annotation string d'une ressource RDF.

        Args:
            uri: URI de la ressource
            annotation_name: Nom de l'annotation (sans préfixe)

        Returns:
            Valeur de l'annotation ou chaîne vide
        """
        if self.namespace:
            annotation_uri = self.namespace[annotation_name]
            for value in self.graph.objects(uri, annotation_uri):
                return str(value)
        return ""

    def get_hub_concepts(self) -> Dict[str, OntologyConcept]:
        """Retourne les concepts marqués comme Hubs."""
        all_concepts = self.get_all_concepts()
        return {uri: c for uri, c in all_concepts.items() if c.is_hub}

    def get_satellite_concepts(self) -> Dict[str, OntologyConcept]:
        """Retourne les concepts marqués comme Satellites."""
        all_concepts = self.get_all_concepts()
        return {uri: c for uri, c in all_concepts.items() if c.is_satellite}

    def get_lemma_mapping(self) -> Dict[str, OntologyConcept]:
        """Retourne un mapping lemma -> concept pour la recherche rapide."""
        all_concepts = self.get_all_concepts()
        return {c.lemma: c for c in all_concepts.values() if c.lemma}

    def get_statistics(self) -> Dict[str, int]:
        """
        Retourne des statistiques sur l'ontologie.

        Returns:
            Dictionnaire avec le nombre de concepts, relations, attributs
        """
        concepts = self.get_all_concepts()
        relations = self.get_all_relations()
        attributes = self.get_all_attributes()

        return {
            "total_triples": len(self.graph),
            "concepts": len(concepts),
            "classes": len([c for c in concepts.values() if c.concept_type == "class"]),
            "individuals": len([c for c in concepts.values() if c.concept_type == "individual"]),
            "relations": len(relations),
            "attributes": len(attributes)
        }
