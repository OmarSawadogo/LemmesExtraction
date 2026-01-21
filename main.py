#!/usr/bin/env python3
"""
Script de lancement du système d'extraction de connaissances.
Point d'entrée principal pour exécution locale.
"""

if __name__ == "__main__":
    from src.app import app
    print("🚀 Lancement de l'application...")
    app.launch()
