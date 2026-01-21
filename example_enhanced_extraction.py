"""
Script d'exemple d'utilisation de l'extracteur amélioré.
Démontre les capacités de validation et de parsing structuré.
"""

import json
from pathlib import Path
from src.enhanced_llm_extractor import EnhancedLLMExtractor, OntologyValidator
from src.config import Config


def example_1_structured_analysis():
    """Exemple 1: Analyse structurée d'une seule image."""
    print("\n" + "="*80)
    print("EXEMPLE 1: ANALYSE STRUCTURÉE D'UNE IMAGE")
    print("="*80)

    # Initialiser l'extracteur en mode structuré
    extractor = EnhancedLLMExtractor(
        ollama_url=Config.OLLAMA_BASE_URL,
        model=Config.LLAVA_MODEL,
        use_structured=True  # Mode structuré (format clé:valeur)
    )

    # Analyser une image
    image_path = Path(Config.IMAGES_PATH) / "1.jpg"

    if not image_path.exists():
        print(f"⚠️  Image non trouvée: {image_path}")
        print("   Veuillez placer des images dans le dossier data/images/")
        return

    print(f"\n📷 Analyse de: {image_path.name}")
    print("-" * 80)

    result = extractor.analyze_plant(str(image_path))

    # Afficher les résultats
    print(f"\n🌱 IDENTIFICATION")
    print(f"   Plante: {result.plante}")
    print(f"   État: {'Malade ❌' if result.est_malade else 'Saine ✅'}")

    if result.est_malade:
        print(f"   Type problème: {result.type_probleme}")
        print(f"   Problème identifié: {result.nom_probleme}")
        print(f"   Symptômes: {', '.join(result.symptomes)}")

    print(f"\n📊 CLASSIFICATION HUB-LINK-SATELLITE")
    print(f"   Hub: {result.hub}")
    print(f"   Link: {result.link}")
    print(f"   Satellites ({len(result.satellites)}): {', '.join(result.satellites)}")

    print(f"\n👁️  OBSERVATIONS")
    print(f"   État feuilles: {result.etat_feuille}")
    print(f"   Couleur: {result.couleur_feuille}")

    print(f"\n🔍 VALIDATION")
    print(f"   Lemmes bruts: {', '.join(result.lemmes_bruts)}")
    print(f"   Lemmes validés: {', '.join(result.lemmes_valides)}")

    if result.lemmes_corriges:
        print(f"\n   ✏️  Corrections appliquées:")
        for original, corrected in result.lemmes_corriges.items():
            print(f"      {original} → {corrected}")

    print(f"\n📈 MÉTADONNÉES")
    print(f"   Score confiance: {result.confidence_score:.2%}")
    print(f"   Timestamp: {result.timestamp}")

    # Exporter en JSON
    output_file = Path("exports") / f"enhanced_analysis_{image_path.stem}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultat exporté: {output_file}")


def example_2_lemma_mode_comparison():
    """Exemple 2: Comparaison mode structuré vs mode lemmes."""
    print("\n" + "="*80)
    print("EXEMPLE 2: COMPARAISON MODE STRUCTURÉ vs MODE LEMMES")
    print("="*80)

    image_path = Path(Config.IMAGES_PATH) / "1.jpg"

    if not image_path.exists():
        print(f"⚠️  Image non trouvée: {image_path}")
        return

    # Mode structuré
    print("\n🔹 MODE STRUCTURÉ (recommandé)")
    print("-" * 80)
    extractor_structured = EnhancedLLMExtractor(
        ollama_url=Config.OLLAMA_BASE_URL,
        model=Config.LLAVA_MODEL,
        use_structured=True
    )
    result_structured = extractor_structured.analyze_plant(str(image_path))

    print(f"   Lemmes validés: {', '.join(result_structured.lemmes_valides)}")
    print(f"   Confiance: {result_structured.confidence_score:.2%}")
    print(f"   Corrections: {len(result_structured.lemmes_corriges)}")

    # Mode lemmes
    print("\n🔹 MODE LEMMES LIBRES")
    print("-" * 80)
    extractor_lemmas = EnhancedLLMExtractor(
        ollama_url=Config.OLLAMA_BASE_URL,
        model=Config.LLAVA_MODEL,
        use_structured=False
    )
    result_lemmas = extractor_lemmas.analyze_plant(str(image_path))

    print(f"   Lemmes validés: {', '.join(result_lemmas.lemmes_valides)}")
    print(f"   Confiance: {result_lemmas.confidence_score:.2%}")
    print(f"   Corrections: {len(result_lemmas.lemmes_corriges)}")

    # Comparaison
    print(f"\n📊 COMPARAISON")
    print(f"   Structuré - Lemmes valides: {len(result_structured.lemmes_valides)}")
    print(f"   Lemmes - Lemmes valides: {len(result_lemmas.lemmes_valides)}")
    print(f"   Structuré - Confiance: {result_structured.confidence_score:.2%}")
    print(f"   Lemmes - Confiance: {result_lemmas.confidence_score:.2%}")


def example_3_batch_analysis():
    """Exemple 3: Analyse par lot avec validation."""
    print("\n" + "="*80)
    print("EXEMPLE 3: ANALYSE PAR LOT")
    print("="*80)

    extractor = EnhancedLLMExtractor(
        ollama_url=Config.OLLAMA_BASE_URL,
        model=Config.LLAVA_MODEL,
        use_structured=True
    )

    # Récupérer les 3 premières images
    images_dir = Path(Config.IMAGES_PATH)
    image_files = sorted(images_dir.glob("*.jpg"))[:3]

    if not image_files:
        print(f"⚠️  Aucune image trouvée dans {images_dir}")
        return

    print(f"\n📸 Analyse de {len(image_files)} images...")
    print("-" * 80)

    results = extractor.batch_analyze([str(img) for img in image_files])

    # Résumé des résultats
    print(f"\n📊 RÉSUMÉ DES ANALYSES")
    print("-" * 80)

    for img_name, analysis in results.items():
        if analysis:
            status = "❌ Malade" if analysis.est_malade else "✅ Saine"
            print(f"\n{img_name}:")
            print(f"   Plante: {analysis.plante} - {status}")
            print(f"   Link: {analysis.link}")
            print(f"   Satellites: {len(analysis.satellites)}")
            print(f"   Confiance: {analysis.confidence_score:.2%}")

            if analysis.lemmes_corriges:
                print(f"   Corrections: {len(analysis.lemmes_corriges)}")
        else:
            print(f"\n{img_name}: ❌ Échec de l'analyse")

    # Statistiques globales
    valid_results = [r for r in results.values() if r]
    if valid_results:
        print(f"\n📈 STATISTIQUES GLOBALES")
        print("-" * 80)
        print(f"   Images analysées: {len(valid_results)}/{len(results)}")
        print(f"   Taux de succès: {len(valid_results)/len(results):.0%}")

        malades = sum(1 for r in valid_results if r.est_malade)
        saines = len(valid_results) - malades
        print(f"   Plantes malades: {malades}")
        print(f"   Plantes saines: {saines}")

        avg_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results)
        print(f"   Confiance moyenne: {avg_confidence:.2%}")

        total_corrections = sum(len(r.lemmes_corriges) for r in valid_results)
        print(f"   Corrections totales: {total_corrections}")


def example_4_validation_demo():
    """Exemple 4: Démonstration des capacités de validation."""
    print("\n" + "="*80)
    print("EXEMPLE 4: DÉMONSTRATION DE VALIDATION")
    print("="*80)

    validator = OntologyValidator()

    # Test de validation de termes
    test_terms = [
        ("maïs", "plante"),
        ("mais", "plante"),
        ("helminthosporiose", "maladie"),
        ("helmintho", "maladie"),  # Partiel
        ("nécrose", "symptôme"),
        ("necros", "symptôme"),  # Fuzzy match
        ("vert_foncé", "couleur"),
        ("vertfonce", "couleur"),  # Sans underscore
        ("chlorotik", "état"),  # Erreur d'orthographe
    ]

    print("\n🔍 TEST DE VALIDATION DES TERMES")
    print("-" * 80)

    for term, category in test_terms:
        # Choisir la liste de validation selon la catégorie
        if category == "plante":
            valid_list = validator.HUBS["plantes"]
            result = validator.identify_plant(term)
        elif category == "maladie":
            # Chercher dans toutes les maladies
            valid_list = []
            for key in validator.SATELLITES:
                if key.startswith("maladies_"):
                    valid_list.extend(validator.SATELLITES[key])
            result = validator.validate_term(term, valid_list)
        elif category == "symptôme":
            valid_list = validator.SATELLITES["symptomes"]
            result = validator.validate_term(term, valid_list)
        elif category == "couleur":
            valid_list = validator.SATELLITES["couleurs_feuille"]
            result = validator.validate_term(term, valid_list)
        elif category == "état":
            valid_list = validator.SATELLITES["etats_feuille"]
            result = validator.validate_term(term, valid_list)
        else:
            result = None

        if result:
            if result == term:
                print(f"   ✅ '{term}' → '{result}' (exact)")
            else:
                print(f"   ✏️  '{term}' → '{result}' (corrigé)")
        else:
            print(f"   ❌ '{term}' → Non reconnu")

    # Statistiques de l'ontologie
    print(f"\n📚 STATISTIQUES DE L'ONTOLOGIE")
    print("-" * 80)
    print(f"   Hubs (plantes): {len(validator.HUBS['plantes'])}")
    print(f"   Links (relations): {len(validator.LINKS['relations'])}")

    total_satellites = sum(len(terms) for terms in validator.SATELLITES.values())
    print(f"   Satellites: {total_satellites}")

    print(f"\n   Détail des satellites:")
    for category, terms in validator.SATELLITES.items():
        print(f"      {category}: {len(terms)} termes")

    all_terms = validator.get_all_valid_terms()
    print(f"\n   Total de termes dans l'ontologie: {len(all_terms)}")


def main():
    """Fonction principale - exécute tous les exemples."""
    print("\n" + "="*80)
    print("🌿 DÉMONSTRATION DE L'EXTRACTEUR AMÉLIORÉ")
    print("   Extraction de lemmes avec validation stricte et parsing structuré")
    print("="*80)

    try:
        # Exemple 1: Analyse structurée de base
        example_1_structured_analysis()

        # Exemple 2: Comparaison des modes
        # example_2_lemma_mode_comparison()

        # Exemple 3: Analyse par lot
        # example_3_batch_analysis()

        # Exemple 4: Démonstration validation
        example_4_validation_demo()

        print("\n" + "="*80)
        print("✅ DÉMONSTRATION TERMINÉE")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
