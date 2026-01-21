"""
Exporteur JSON pour schémas Data Vault.
Sérialise le schéma en format JSON structuré.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
from src.datavault_generator import DataVaultSchema


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour gérer les types numpy."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class JSONExporter:
    """Exporteur JSON pour Data Vault."""

    def export(self, schema: DataVaultSchema, output_path: str) -> str:
        """
        Exporte le schéma en format JSON.

        Args:
            schema: Schéma Data Vault à exporter
            output_path: Chemin du fichier de sortie

        Returns:
            Chemin du fichier créé
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convertir le schéma en dictionnaire
        schema_dict = schema.to_dict()

        # Ajouter des métadonnées d'export
        schema_dict["export_metadata"] = {
            "format": "json",
            "exported_at": datetime.now().isoformat(),
            "exporter_version": "1.0.0"
        }

        # Écrire le fichier JSON avec indentation et encodeur numpy
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"📄 Export JSON réussi: {output_path}")
        return str(output_path)

    def export_compact(self, schema: DataVaultSchema, output_path: str) -> str:
        """
        Exporte le schéma en format JSON compact (sans indentation).

        Args:
            schema: Schéma Data Vault à exporter
            output_path: Chemin du fichier de sortie

        Returns:
            Chemin du fichier créé
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        schema_dict = schema.to_dict()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, ensure_ascii=False, cls=NumpyEncoder)

        print(f"📄 Export JSON compact réussi: {output_path}")
        return str(output_path)
