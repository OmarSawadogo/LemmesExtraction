"""
Modèle Hub pour Data Vault.
Les Hubs représentent les entités métier (plantes, maladies, symptômes).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class Hub:
    """
    Représente une entité métier dans le schéma Data Vault.

    Attributes:
        business_key: Identifiant métier unique (ex: "mais", "helminthosporiose")
        entity_type: Type d'entité (ex: "plante", "maladie", "symptome")
        load_date: Date et heure de chargement
        record_source: Source de l'enregistrement (ex: "img1.jpg")
        ontology_uri: URI de l'entité dans l'ontologie
        confidence_score: Score de similarité avec le concept ontologique [0,1]
        hub_key: Clé technique générée (hash ou UUID)
    """

    business_key: str
    entity_type: str
    record_source: str
    ontology_uri: str
    confidence_score: float
    load_date: datetime = field(default_factory=datetime.now)
    hub_key: Optional[str] = None

    def __post_init__(self):
        """Génère automatiquement la clé hub si non fournie."""
        if self.hub_key is None:
            self.hub_key = self._generate_hub_key()

    def _generate_hub_key(self) -> str:
        """
        Génère une clé unique pour le hub basée sur business_key et entity_type.

        Returns:
            str: Clé hub (hash MD5)
        """
        key_string = f"{self.business_key}_{self.entity_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def to_dict(self) -> dict:
        """
        Convertit le Hub en dictionnaire pour export.

        Returns:
            dict: Représentation dictionnaire du Hub
        """
        return {
            "hub_key": self.hub_key,
            "business_key": self.business_key,
            "entity_type": self.entity_type,
            "ontology_uri": self.ontology_uri,
            "confidence_score": round(self.confidence_score, 4),
            "load_date": self.load_date.isoformat(),
            "record_source": self.record_source
        }

    def __repr__(self) -> str:
        """Représentation string du Hub."""
        return (f"Hub(business_key='{self.business_key}', "
                f"entity_type='{self.entity_type}', "
                f"confidence={self.confidence_score:.2f})")
