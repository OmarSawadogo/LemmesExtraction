"""
Module de génération et validation du schéma Data Vault.
Assemble les Hubs, Links et Satellites en un schéma cohérent.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
from src.models.hub import Hub
from src.models.link import Link
from src.models.satellite import Satellite


@dataclass
class DataVaultSchema:
    """
    Schéma Data Vault complet.

    Attributes:
        hubs: Liste des Hubs (entités)
        links: Liste des Links (relations)
        satellites: Liste des Satellites (attributs)
        metadata: Métadonnées du schéma
    """
    hubs: List[Hub] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    satellites: List[Satellite] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convertit le schéma en dictionnaire pour export.

        Returns:
            Dictionnaire complet du schéma
        """
        return {
            "metadata": self.metadata,
            "hubs": [hub.to_dict() for hub in self.hubs],
            "links": [link.to_dict() for link in self.links],
            "satellites": [satellite.to_dict() for satellite in self.satellites]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur le schéma.

        Returns:
            Dictionnaire de statistiques
        """
        # Compter les entités par type
        entity_types = {}
        for hub in self.hubs:
            entity_types[hub.entity_type] = entity_types.get(hub.entity_type, 0) + 1

        # Compter les relations par type
        relation_types = {}
        for link in self.links:
            relation_types[link.relation_type] = relation_types.get(link.relation_type, 0) + 1

        # Compter les attributs par nom
        attribute_names = {}
        for satellite in self.satellites:
            attribute_names[satellite.attribute_name] = attribute_names.get(satellite.attribute_name, 0) + 1

        # Calculer scores moyens
        avg_hub_score = sum(h.confidence_score for h in self.hubs) / len(self.hubs) if self.hubs else 0
        avg_link_score = sum(l.confidence_score for l in self.links) / len(self.links) if self.links else 0
        avg_sat_score = sum(s.confidence_score for s in self.satellites) / len(self.satellites) if self.satellites else 0

        return {
            "total_hubs": len(self.hubs),
            "total_links": len(self.links),
            "total_satellites": len(self.satellites),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "attribute_names": attribute_names,
            "average_confidence": {
                "hubs": round(avg_hub_score, 3),
                "links": round(avg_link_score, 3),
                "satellites": round(avg_sat_score, 3)
            }
        }


class DataVaultGenerator:
    """
    Générateur et validateur de schémas Data Vault.
    """

    def __init__(self):
        """Initialise le générateur."""
        pass

    def generate_schema(
        self,
        hubs: List[Hub],
        links: List[Link],
        satellites: List[Satellite],
        source_image: str,
        lemmas: List[str]
    ) -> DataVaultSchema:
        """
        Génère un schéma Data Vault à partir des composants.

        Args:
            hubs: Liste des Hubs
            links: Liste des Links
            satellites: Liste des Satellites
            source_image: Nom de l'image source
            lemmas: Lemmes extraits originaux

        Returns:
            Schéma Data Vault complet
        """
        # Créer métadonnées
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "source_image": source_image,
            "original_lemmas": lemmas,
            "lemmas_count": len(lemmas),
            "generator_version": "1.0.0"
        }

        # Créer le schéma
        schema = DataVaultSchema(
            hubs=hubs,
            links=links,
            satellites=satellites,
            metadata=metadata
        )

        print(f"✅ Schéma Data Vault généré: "
              f"{len(hubs)} Hubs, {len(links)} Links, {len(satellites)} Satellites")

        return schema

    def validate_schema(self, schema: DataVaultSchema) -> List[str]:
        """
        Valide l'intégrité du schéma Data Vault.

        Args:
            schema: Schéma à valider

        Returns:
            Liste d'erreurs/warnings (vide si tout est valide)
        """
        errors = []

        # 1. Vérifier unicité des clés
        errors.extend(self._check_key_uniqueness(schema))

        # 2. Vérifier intégrité référentielle
        errors.extend(self._check_referential_integrity(schema))

        # 3. Vérifier scores de confiance
        errors.extend(self._check_confidence_scores(schema))

        # 4. Vérifier contraintes structurelles
        errors.extend(self._check_structural_constraints(schema))

        if errors:
            print(f"⚠️  Validation: {len(errors)} problème(s) détecté(s)")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ Schéma valide: toutes les vérifications passées")

        return errors

    def _check_key_uniqueness(self, schema: DataVaultSchema) -> List[str]:
        """Vérifie l'unicité des clés."""
        errors = []

        # Vérifier Hubs
        hub_keys = [h.hub_key for h in schema.hubs]
        if len(hub_keys) != len(set(hub_keys)):
            errors.append("ERREUR: Clés Hub en double détectées")

        # Vérifier Links
        link_keys = [l.link_key for l in schema.links]
        if len(link_keys) != len(set(link_keys)):
            errors.append("ERREUR: Clés Link en double détectées")

        # Vérifier Satellites
        satellite_keys = [s.satellite_key for s in schema.satellites]
        if len(satellite_keys) != len(set(satellite_keys)):
            errors.append("ERREUR: Clés Satellite en double détectées")

        return errors

    def _check_referential_integrity(self, schema: DataVaultSchema) -> List[str]:
        """Vérifie l'intégrité référentielle."""
        errors = []

        # Créer un ensemble des clés hub valides
        valid_hub_keys = {h.hub_key for h in schema.hubs}

        # Vérifier que tous les Links référencent des Hubs existants
        for link in schema.links:
            if link.hub_source_key not in valid_hub_keys:
                errors.append(f"WARNING: Link {link.link_key[:8]}... "
                            f"référence un Hub source invalide")

            if link.hub_target_key not in valid_hub_keys:
                errors.append(f"WARNING: Link {link.link_key[:8]}... "
                            f"référence un Hub target invalide")

        # Vérifier que tous les Satellites référencent des Hubs existants
        for satellite in schema.satellites:
            if satellite.hub_key not in valid_hub_keys:
                errors.append(f"WARNING: Satellite {satellite.satellite_key[:8]}... "
                            f"référence un Hub invalide")

        return errors

    def _check_confidence_scores(self, schema: DataVaultSchema) -> List[str]:
        """Vérifie la validité des scores de confiance."""
        errors = []

        # Vérifier Hubs
        for hub in schema.hubs:
            if not 0 <= hub.confidence_score <= 1:
                errors.append(f"WARNING: Hub {hub.business_key} a un score invalide: "
                            f"{hub.confidence_score}")

        # Vérifier Links
        for link in schema.links:
            if not 0 <= link.confidence_score <= 1:
                errors.append(f"WARNING: Link {link.link_key[:8]}... a un score invalide: "
                            f"{link.confidence_score}")

        # Vérifier Satellites
        for satellite in schema.satellites:
            if not 0 <= satellite.confidence_score <= 1:
                errors.append(f"WARNING: Satellite {satellite.attribute_name} a un score invalide: "
                            f"{satellite.confidence_score}")

        return errors

    def _check_structural_constraints(self, schema: DataVaultSchema) -> List[str]:
        """
        Vérifie les contraintes structurelles du schéma Data Vault.

        Contraintes attendues avec la nouvelle logique:
        - Au moins 1 Hub (la plante)
        - Au maximum 2 Hubs (plante + maladie/ravageur)
        - 0 ou 1 Link (relation plante-problème)
        - Satellites >= 0 (attributs descriptifs)
        """
        errors = []

        num_hubs = len(schema.hubs)
        num_links = len(schema.links)
        num_satellites = len(schema.satellites)

        # Contrainte 1: Au moins 1 Hub requis
        if num_hubs == 0:
            errors.append("WARNING: Aucun Hub détecté. Au moins une plante devrait être identifiée.")

        # Contrainte 2: Cohérence Link-Hub
        # Un Link nécessite au moins 2 Hubs (source et target)
        if num_links > 0 and num_hubs < 2:
            errors.append(f"WARNING: {num_links} Link(s) détecté(s) mais seulement {num_hubs} Hub(s). "
                        f"Un Link nécessite 2 Hubs.")

        # Contrainte 3: Maximum 1 Link entre 2 Hubs spécifiques
        if num_links > 1:
            link_pairs = set()
            for link in schema.links:
                pair = (link.hub_source_key, link.hub_target_key)
                if pair in link_pairs:
                    errors.append(f"WARNING: Links en double détectés entre les mêmes Hubs.")
                    break
                link_pairs.add(pair)

        return errors

    def merge_schemas(self, schemas: List[DataVaultSchema]) -> DataVaultSchema:
        """
        Fusionne plusieurs schémas Data Vault en un seul.

        Args:
            schemas: Liste de schémas à fusionner

        Returns:
            Schéma fusionné
        """
        merged_hubs = []
        merged_links = []
        merged_satellites = []

        # Utiliser des sets pour dédupliquer par clé
        seen_hub_keys = set()
        seen_link_keys = set()
        seen_satellite_keys = set()

        for schema in schemas:
            # Fusionner Hubs
            for hub in schema.hubs:
                if hub.hub_key not in seen_hub_keys:
                    merged_hubs.append(hub)
                    seen_hub_keys.add(hub.hub_key)

            # Fusionner Links
            for link in schema.links:
                if link.link_key not in seen_link_keys:
                    merged_links.append(link)
                    seen_link_keys.add(link.link_key)

            # Fusionner Satellites (garder tous pour historique)
            for satellite in schema.satellites:
                merged_satellites.append(satellite)
                seen_satellite_keys.add(satellite.satellite_key)

        # Créer métadonnées fusionnées
        merged_metadata = {
            "generated_at": datetime.now().isoformat(),
            "merged_from": len(schemas),
            "source_images": [s.metadata.get("source_image", "unknown") for s in schemas],
            "generator_version": "1.0.0"
        }

        merged_schema = DataVaultSchema(
            hubs=merged_hubs,
            links=merged_links,
            satellites=merged_satellites,
            metadata=merged_metadata
        )

        print(f"🔗 {len(schemas)} schémas fusionnés: "
              f"{len(merged_hubs)} Hubs, {len(merged_links)} Links, "
              f"{len(merged_satellites)} Satellites")

        return merged_schema
