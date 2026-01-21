"""
Modèle Satellite pour Data Vault.
Les Satellites représentent les attributs descriptifs des entités (Hubs).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class Satellite:
    """
    Représente un attribut descriptif d'une entité dans le schéma Data Vault.

    Attributes:
        hub_key: Clé du Hub parent
        attribute_name: Nom de l'attribut (ex: "couleur_feuille", "forme_feuille")
        attribute_value: Valeur de l'attribut (ex: "vert_moyen", "lineaire_lanceolee")
        record_source: Source de l'enregistrement (ex: "img1.jpg")
        confidence_score: Score de confiance de l'attribut [0,1]
        load_date: Date et heure de chargement
        satellite_key: Clé technique générée
        hash_diff: Hash pour détection de changements
    """

    hub_key: str
    attribute_name: str
    attribute_value: str
    record_source: str
    confidence_score: float
    load_date: datetime = field(default_factory=datetime.now)
    satellite_key: Optional[str] = None
    hash_diff: Optional[str] = None

    def __post_init__(self):
        """Génère automatiquement les clés si non fournies."""
        if self.satellite_key is None:
            self.satellite_key = self._generate_satellite_key()
        if self.hash_diff is None:
            self.hash_diff = self._generate_hash_diff()

    def _generate_satellite_key(self) -> str:
        """
        Génère une clé unique pour le satellite.

        Returns:
            str: Clé satellite (hash MD5)
        """
        key_string = f"{self.hub_key}_{self.attribute_name}_{self.load_date.timestamp()}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_hash_diff(self) -> str:
        """
        Génère un hash pour détecter les changements d'attributs.

        Returns:
            str: Hash des attributs (MD5)
        """
        diff_string = f"{self.attribute_name}_{self.attribute_value}_{self.confidence_score}"
        return hashlib.md5(diff_string.encode()).hexdigest()

    def to_dict(self) -> dict:
        """
        Convertit le Satellite en dictionnaire pour export.

        Returns:
            dict: Représentation dictionnaire du Satellite
        """
        return {
            "satellite_key": self.satellite_key,
            "hub_key": self.hub_key,
            "attribute_name": self.attribute_name,
            "attribute_value": self.attribute_value,
            "confidence_score": round(self.confidence_score, 4),
            "load_date": self.load_date.isoformat(),
            "record_source": self.record_source,
            "hash_diff": self.hash_diff
        }

    def __repr__(self) -> str:
        """Représentation string du Satellite."""
        return (f"Satellite({self.attribute_name}='{self.attribute_value}', "
                f"hub={self.hub_key[:8]}...)")
