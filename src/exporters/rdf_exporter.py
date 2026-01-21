"""
Exporteur RDF/Turtle pour schémas Data Vault.
Génère des triplets RDF à partir du schéma.
"""

from pathlib import Path
from datetime import datetime
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, XSD
from src.datavault_generator import DataVaultSchema


class RDFExporter:
    """Exporteur RDF/Turtle pour Data Vault."""

    def __init__(self, base_uri: str = "http://www.example.org/datavault/"):
        """
        Initialise l'exporteur RDF.

        Args:
            base_uri: URI de base pour les ressources
        """
        self.base_uri = base_uri
        self.ns = Namespace(base_uri)
        self.dv_ns = Namespace(base_uri + "schema/")

    def export(self, schema: DataVaultSchema, output_path: str, format: str = "turtle") -> str:
        """
        Exporte le schéma en format RDF.

        Args:
            schema: Schéma Data Vault à exporter
            output_path: Chemin du fichier de sortie
            format: Format RDF ("turtle", "xml", "n3", "nt")

        Returns:
            Chemin du fichier créé
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Créer le graphe RDF
        g = Graph()
        g.bind("dv", self.ns)
        g.bind("dvschema", self.dv_ns)

        # Ajouter les triplets pour les Hubs
        for hub in schema.hubs:
            hub_uri = URIRef(self.ns[f"hub/{hub.hub_key}"])

            g.add((hub_uri, RDF.type, self.dv_ns.Hub))
            g.add((hub_uri, self.dv_ns.businessKey, Literal(hub.business_key)))
            g.add((hub_uri, self.dv_ns.entityType, Literal(hub.entity_type)))
            g.add((hub_uri, self.dv_ns.ontologyURI, URIRef(hub.ontology_uri)))
            g.add((hub_uri, self.dv_ns.confidenceScore, Literal(hub.confidence_score, datatype=XSD.float)))
            g.add((hub_uri, self.dv_ns.loadDate, Literal(hub.load_date.isoformat(), datatype=XSD.dateTime)))
            g.add((hub_uri, self.dv_ns.recordSource, Literal(hub.record_source)))

        # Ajouter les triplets pour les Links
        for link in schema.links:
            link_uri = URIRef(self.ns[f"link/{link.link_key}"])
            source_uri = URIRef(self.ns[f"hub/{link.hub_source_key}"])
            target_uri = URIRef(self.ns[f"hub/{link.hub_target_key}"])

            g.add((link_uri, RDF.type, self.dv_ns.Link))
            g.add((link_uri, self.dv_ns.relationType, Literal(link.relation_type)))
            g.add((link_uri, self.dv_ns.hubSource, source_uri))
            g.add((link_uri, self.dv_ns.hubTarget, target_uri))
            g.add((link_uri, self.dv_ns.confidenceScore, Literal(link.confidence_score, datatype=XSD.float)))
            g.add((link_uri, self.dv_ns.loadDate, Literal(link.load_date.isoformat(), datatype=XSD.dateTime)))
            g.add((link_uri, self.dv_ns.recordSource, Literal(link.record_source)))

        # Ajouter les triplets pour les Satellites
        for satellite in schema.satellites:
            sat_uri = URIRef(self.ns[f"satellite/{satellite.satellite_key}"])
            hub_uri = URIRef(self.ns[f"hub/{satellite.hub_key}"])

            g.add((sat_uri, RDF.type, self.dv_ns.Satellite))
            g.add((sat_uri, self.dv_ns.hubKey, hub_uri))
            g.add((sat_uri, self.dv_ns.attributeName, Literal(satellite.attribute_name)))
            g.add((sat_uri, self.dv_ns.attributeValue, Literal(satellite.attribute_value)))
            g.add((sat_uri, self.dv_ns.confidenceScore, Literal(satellite.confidence_score, datatype=XSD.float)))
            g.add((sat_uri, self.dv_ns.loadDate, Literal(satellite.load_date.isoformat(), datatype=XSD.dateTime)))
            g.add((sat_uri, self.dv_ns.recordSource, Literal(satellite.record_source)))
            g.add((sat_uri, self.dv_ns.hashDiff, Literal(satellite.hash_diff)))

        # Ajouter métadonnées du schéma
        schema_uri = URIRef(self.ns["schema"])
        g.add((schema_uri, RDF.type, self.dv_ns.DataVaultSchema))
        g.add((schema_uri, self.dv_ns.exportedAt, Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))
        g.add((schema_uri, self.dv_ns.totalHubs, Literal(len(schema.hubs), datatype=XSD.integer)))
        g.add((schema_uri, self.dv_ns.totalLinks, Literal(len(schema.links), datatype=XSD.integer)))
        g.add((schema_uri, self.dv_ns.totalSatellites, Literal(len(schema.satellites), datatype=XSD.integer)))

        # Sérialiser le graphe
        g.serialize(destination=str(output_path), format=format, encoding='utf-8')

        print(f"🔗 Export RDF/{format} réussi: {output_path}")
        print(f"   {len(g)} triplets générés")

        return str(output_path)
