"""
Exporteur SQL pour schémas Data Vault.
Génère des scripts SQL (CREATE TABLE + INSERT) pour PostgreSQL.
"""

from pathlib import Path
from datetime import datetime
from src.datavault_generator import DataVaultSchema


class SQLExporter:
    """Exporteur SQL pour Data Vault (dialecte PostgreSQL)."""

    def export(self, schema: DataVaultSchema, output_path: str) -> str:
        """
        Exporte le schéma en format SQL.

        Args:
            schema: Schéma Data Vault à exporter
            output_path: Chemin du fichier de sortie

        Returns:
            Chemin du fichier créé
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sql_statements = []

        # En-tête
        sql_statements.append(self._generate_header(schema))

        # CREATE TABLE statements
        sql_statements.append(self._generate_create_tables())

        # INSERT statements pour Hubs
        sql_statements.append(self._generate_hub_inserts(schema.hubs))

        # INSERT statements pour Links
        sql_statements.append(self._generate_link_inserts(schema.links))

        # INSERT statements pour Satellites
        sql_statements.append(self._generate_satellite_inserts(schema.satellites))

        # Créer indexes
        sql_statements.append(self._generate_indexes())

        # Écrire le fichier SQL
        sql_content = "\n\n".join(sql_statements)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sql_content)

        print(f"💾 Export SQL réussi: {output_path}")
        return str(output_path)

    def _generate_header(self, schema: DataVaultSchema) -> str:
        """Génère l'en-tête SQL."""
        return f"""-- Data Vault Schema Export
-- Generated at: {datetime.now().isoformat()}
-- Source: {schema.metadata.get('source_image', 'unknown')}
-- Statistics: {len(schema.hubs)} Hubs, {len(schema.links)} Links, {len(schema.satellites)} Satellites
-- Database: PostgreSQL

-- Drop existing tables if needed
-- DROP TABLE IF EXISTS dv_satellites CASCADE;
-- DROP TABLE IF EXISTS dv_links CASCADE;
-- DROP TABLE IF EXISTS dv_hubs CASCADE;"""

    def _generate_create_tables(self) -> str:
        """Génère les CREATE TABLE statements."""
        return """-- ============================================================
-- CREATE TABLES
-- ============================================================

-- Table Hubs (Entités)
CREATE TABLE IF NOT EXISTS dv_hubs (
    hub_key VARCHAR(32) PRIMARY KEY,
    business_key VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    ontology_uri TEXT,
    confidence_score NUMERIC(5, 4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    load_date TIMESTAMP NOT NULL,
    record_source VARCHAR(255) NOT NULL,
    UNIQUE(business_key, entity_type)
);

-- Table Links (Relations)
CREATE TABLE IF NOT EXISTS dv_links (
    link_key VARCHAR(32) PRIMARY KEY,
    hub_source_key VARCHAR(32) NOT NULL REFERENCES dv_hubs(hub_key),
    hub_target_key VARCHAR(32) NOT NULL REFERENCES dv_hubs(hub_key),
    relation_type VARCHAR(255) NOT NULL,
    confidence_score NUMERIC(5, 4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    load_date TIMESTAMP NOT NULL,
    record_source VARCHAR(255) NOT NULL
);

-- Table Satellites (Attributs)
CREATE TABLE IF NOT EXISTS dv_satellites (
    satellite_key VARCHAR(32) PRIMARY KEY,
    hub_key VARCHAR(32) NOT NULL REFERENCES dv_hubs(hub_key),
    attribute_name VARCHAR(255) NOT NULL,
    attribute_value TEXT NOT NULL,
    confidence_score NUMERIC(5, 4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    load_date TIMESTAMP NOT NULL,
    record_source VARCHAR(255) NOT NULL,
    hash_diff VARCHAR(32) NOT NULL
);"""

    def _generate_hub_inserts(self, hubs) -> str:
        """Génère les INSERT statements pour les Hubs."""
        if not hubs:
            return "-- No Hubs to insert"

        statements = ["-- ============================================================",
                     "-- INSERT HUBS",
                     "-- ============================================================",
                     ""]

        for hub in hubs:
            sql = f"""INSERT INTO dv_hubs (hub_key, business_key, entity_type, ontology_uri, confidence_score, load_date, record_source)
VALUES ('{hub.hub_key}', '{self._escape_sql(hub.business_key)}', '{hub.entity_type}',
        '{hub.ontology_uri}', {hub.confidence_score:.4f},
        '{hub.load_date.isoformat()}', '{self._escape_sql(hub.record_source)}');"""
            statements.append(sql)

        return "\n".join(statements)

    def _generate_link_inserts(self, links) -> str:
        """Génère les INSERT statements pour les Links."""
        if not links:
            return "-- No Links to insert"

        statements = ["-- ============================================================",
                     "-- INSERT LINKS",
                     "-- ============================================================",
                     ""]

        for link in links:
            sql = f"""INSERT INTO dv_links (link_key, hub_source_key, hub_target_key, relation_type, confidence_score, load_date, record_source)
VALUES ('{link.link_key}', '{link.hub_source_key}', '{link.hub_target_key}',
        '{self._escape_sql(link.relation_type)}', {link.confidence_score:.4f},
        '{link.load_date.isoformat()}', '{self._escape_sql(link.record_source)}');"""
            statements.append(sql)

        return "\n".join(statements)

    def _generate_satellite_inserts(self, satellites) -> str:
        """Génère les INSERT statements pour les Satellites."""
        if not satellites:
            return "-- No Satellites to insert"

        statements = ["-- ============================================================",
                     "-- INSERT SATELLITES",
                     "-- ============================================================",
                     ""]

        for satellite in satellites:
            sql = f"""INSERT INTO dv_satellites (satellite_key, hub_key, attribute_name, attribute_value, confidence_score, load_date, record_source, hash_diff)
VALUES ('{satellite.satellite_key}', '{satellite.hub_key}',
        '{self._escape_sql(satellite.attribute_name)}', '{self._escape_sql(satellite.attribute_value)}',
        {satellite.confidence_score:.4f}, '{satellite.load_date.isoformat()}',
        '{self._escape_sql(satellite.record_source)}', '{satellite.hash_diff}');"""
            statements.append(sql)

        return "\n".join(statements)

    def _generate_indexes(self) -> str:
        """Génère les CREATE INDEX statements."""
        return """-- ============================================================
-- CREATE INDEXES
-- ============================================================

-- Indexes pour Hubs
CREATE INDEX IF NOT EXISTS idx_hubs_business_key ON dv_hubs(business_key);
CREATE INDEX IF NOT EXISTS idx_hubs_entity_type ON dv_hubs(entity_type);
CREATE INDEX IF NOT EXISTS idx_hubs_load_date ON dv_hubs(load_date);

-- Indexes pour Links
CREATE INDEX IF NOT EXISTS idx_links_source ON dv_links(hub_source_key);
CREATE INDEX IF NOT EXISTS idx_links_target ON dv_links(hub_target_key);
CREATE INDEX IF NOT EXISTS idx_links_relation ON dv_links(relation_type);
CREATE INDEX IF NOT EXISTS idx_links_load_date ON dv_links(load_date);

-- Indexes pour Satellites
CREATE INDEX IF NOT EXISTS idx_satellites_hub ON dv_satellites(hub_key);
CREATE INDEX IF NOT EXISTS idx_satellites_attribute ON dv_satellites(attribute_name);
CREATE INDEX IF NOT EXISTS idx_satellites_load_date ON dv_satellites(load_date);
CREATE INDEX IF NOT EXISTS idx_satellites_hash_diff ON dv_satellites(hash_diff);

-- ============================================================
-- QUERIES EXAMPLES
-- ============================================================

-- Voir tous les Hubs avec leurs Satellites
-- SELECT h.business_key, h.entity_type, s.attribute_name, s.attribute_value
-- FROM dv_hubs h
-- LEFT JOIN dv_satellites s ON h.hub_key = s.hub_key
-- ORDER BY h.business_key, s.attribute_name;

-- Voir toutes les relations
-- SELECT h1.business_key as source, l.relation_type, h2.business_key as target
-- FROM dv_links l
-- JOIN dv_hubs h1 ON l.hub_source_key = h1.hub_key
-- JOIN dv_hubs h2 ON l.hub_target_key = h2.hub_key;"""

    @staticmethod
    def _escape_sql(value: str) -> str:
        """Échappe les caractères spéciaux SQL."""
        return value.replace("'", "''")
