"""
Modèle Link pour Data Vault.
Les Links représentent les relations entre entités (Hubs).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class Link:
    """
    Représente une relation entre deux entités dans le schéma Data Vault.

    Attributes:
        hub_source_key: Clé du Hub source
        hub_target_key: Clé du Hub cible
        relation_type: Type de relation (ex: "a_maladie_mais", "presente_symptome")
        record_source: Source de l'enregistrement (ex: "img1.jpg")
        confidence_score: Score de confiance de la relation [0,1]
        load_date: Date et heure de chargement
        link_key: Clé technique générée
    """

    hub_source_key: str
    hub_target_key: str
    relation_type: str
    record_source: str
    confidence_score: float
    load_date: datetime = field(default_factory=datetime.now)
    link_key: Optional[str] = None

    def __post_init__(self):
        """Génère automatiquement la clé link si non fournie."""
        if self.link_key is None:
            self.link_key = self._generate_link_key()

    def _generate_link_key(self) -> str:
        """
        Génère une clé unique pour le link basée sur les hubs source/target et la relation.

        Returns:
            str: Clé link (hash MD5)
        """
        key_string = f"{self.hub_source_key}_{self.relation_type}_{self.hub_target_key}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def to_dict(self) -> dict:
        """
        Convertit le Link en dictionnaire pour export.

        Returns:
            dict: Représentation dictionnaire du Link
        """
        return {
            "link_key": self.link_key,
            "hub_source_key": self.hub_source_key,
            "hub_target_key": self.hub_target_key,
            "relation_type": self.relation_type,
            "confidence_score": round(self.confidence_score, 4),
            "load_date": self.load_date.isoformat(),
            "record_source": self.record_source
        }

    def __repr__(self) -> str:
        """Représentation string du Link."""
        return (f"Link({self.hub_source_key[:8]}... "
                f"--[{self.relation_type}]--> "
                f"{self.hub_target_key[:8]}...)")
