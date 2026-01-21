"""
Module de calcul de similarité entre lemmes et termes ontologiques.
Combine similarité lexicale (Jaro-Winkler) et sémantique (embeddings).
Supporte l'accélération GPU via CuPy si disponible.
"""

from typing import Dict, List
from collections import Counter
from functools import lru_cache
import numpy as np
from unidecode import unidecode
import jellyfish
import requests

# Tentative d'importation de CuPy pour l'accélération GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


@lru_cache(maxsize=1024)
def _normalize_cached(s: str) -> str:
    """Normalise une chaîne avec cache LRU."""
    s = s.lower()
    s = unidecode(s)
    s = s.replace('_', ' ').replace('-', ' ')
    return ' '.join(s.split()).strip()


@lru_cache(maxsize=512)
def _get_ngrams_cached(text: str, n: int = 2) -> tuple:
    """Génère les n-grams avec cache LRU."""
    padded = f"${text}$"
    return tuple(padded[i:i+n] for i in range(len(padded) - n + 1))


class SimilarityCalculator:
    """
    Calculateur de similarité hybride (lexicale + sémantique).
    Utilise Ollama pour les embeddings.
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text:latest",
        ollama_base_url: str = "http://host.docker.internal:11434",
        algorithm: str = "hybrid",
        use_gpu: bool = True
    ):
        """
        Initialise le calculateur de similarité.

        Args:
            embedding_model: Nom du modèle d'embeddings dans Ollama
            ollama_base_url: URL de base d'Ollama
            algorithm: Algorithme à utiliser ("lexical", "semantic", "hybrid", "jaro_winkler", "cosine", "jaro_cosine")
            use_gpu: Utiliser le GPU si disponible (CuPy)
        """
        self.embedding_model_name = embedding_model
        self.ollama_base_url = ollama_base_url
        self.embeddings_endpoint = f"{ollama_base_url}/api/embeddings"
        self.algorithm = algorithm
        self.use_gpu = use_gpu and GPU_AVAILABLE

        print(f"[EMB] Modele d'embeddings Ollama: {embedding_model}")
        print(f"[ALG] Algorithme de similarite: {algorithm}")

        # Afficher le statut GPU
        if self.use_gpu:
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = gpu_info['name'].decode('utf-8') if isinstance(gpu_info['name'], bytes) else gpu_info['name']
            print(f"[GPU] Acceleration GPU activee: {gpu_name}")
        elif use_gpu and not GPU_AVAILABLE:
            print("[GPU] CuPy non disponible - installation: pip install cupy-cuda12x")
            print("[CPU] Utilisation du CPU pour les calculs")
        else:
            print("[CPU] Utilisation du CPU pour les calculs")

        # Vérifier la connexion à Ollama
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("[OK] Connexion a Ollama reussie")
            else:
                print("[!] Impossible de se connecter a Ollama")
        except Exception as e:
            print(f"[!] Erreur de connexion a Ollama: {e}")

        # Cache pour les embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Cache GPU pour les embeddings (si GPU disponible)
        self._gpu_embedding_cache: Dict[str, 'cp.ndarray'] = {} if self.use_gpu else {}

    def calculate_similarity(self, lemma: str, ontology_term: str) -> float:
        """
        Calcule la similarité entre un lemme et un terme ontologique.

        Args:
            lemma: Lemme extrait par LLaVA
            ontology_term: Terme de l'ontologie

        Returns:
            Score de similarité [0, 1]
        """
        # Normaliser les chaînes
        lemma_norm = self._normalize_string(lemma)
        term_norm = self._normalize_string(ontology_term)

        # Correspondance exacte = 1.0
        if lemma_norm == term_norm:
            return 1.0

        # Calculer selon l'algorithme sélectionné
        if self.algorithm == "lexical":
            # Similarité lexicale uniquement (Jaro-Winkler avec bonus)
            return self._lexical_similarity(lemma_norm, term_norm)

        elif self.algorithm == "semantic":
            # Similarité sémantique uniquement (embeddings)
            return self._semantic_similarity(lemma, ontology_term)

        elif self.algorithm == "jaro_winkler":
            # Similarité Jaro-Winkler pure (sans bonus)
            return self._jaro_winkler_similarity(lemma_norm, term_norm)

        elif self.algorithm == "cosine":
            # Similarité cosinus basée sur n-grams de caractères
            return self._cosine_ngram_similarity(lemma_norm, term_norm)

        elif self.algorithm == "jaro_cosine":
            # Combinaison Jaro-Winkler + Cosinus (n-grams)
            sim_jw = self._jaro_winkler_similarity(lemma_norm, term_norm)
            sim_cos = self._cosine_ngram_similarity(lemma_norm, term_norm)
            return max(sim_jw, sim_cos)

        else:  # "hybrid" par défaut
            # Hybride: maximum des deux
            sim_lex = self._lexical_similarity(lemma_norm, term_norm)
            sim_sem = self._semantic_similarity(lemma, ontology_term)
            return max(sim_lex, sim_sem)

    def _lexical_similarity(self, s1: str, s2: str) -> float:
        """
        Calcule la similarité lexicale avec Jaro-Winkler.

        Args:
            s1: Première chaîne (normalisée)
            s2: Deuxième chaîne (normalisée)

        Returns:
            Score de similarité lexicale [0, 1]
        """
        if not s1 or not s2:
            return 0.0

        # Calculer Jaro-Winkler
        jw_score = jellyfish.jaro_winkler_similarity(s1, s2)

        # Bonus pour préfixe commun (vérifie longueur minimale)
        if len(s1) >= 3 and len(s2) >= 3:
            if s1.startswith(s2[:3]) or s2.startswith(s1[:3]):
                jw_score = min(1.0, jw_score * 1.1)

        # Bonus pour inclusion
        if s1 in s2 or s2 in s1:
            jw_score = min(1.0, jw_score * 1.15)

        return jw_score

    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """
        Calcule la similarité Jaro-Winkler pure (sans bonus).

        Args:
            s1: Première chaîne (normalisée)
            s2: Deuxième chaîne (normalisée)

        Returns:
            Score de similarité Jaro-Winkler [0, 1]
        """
        if not s1 or not s2:
            return 0.0

        return jellyfish.jaro_winkler_similarity(s1, s2)

    def _cosine_ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        """
        Calcule la similarité cosinus basée sur les n-grams de caractères.

        Args:
            s1: Première chaîne (normalisée)
            s2: Deuxième chaîne (normalisée)
            n: Taille des n-grams (défaut: 2 pour bigrammes)

        Returns:
            Score de similarité cosinus [0, 1]
        """
        if not s1 or not s2:
            return 0.0

        # Générer les n-grams (avec cache)
        ngrams1 = _get_ngrams_cached(s1, n)
        ngrams2 = _get_ngrams_cached(s2, n)

        # Compter les occurrences
        counter1 = Counter(ngrams1)
        counter2 = Counter(ngrams2)

        # Ensemble de tous les n-grams
        all_ngrams = set(counter1.keys()) | set(counter2.keys())

        # Créer les vecteurs
        vec1 = np.array([counter1.get(ng, 0) for ng in all_ngrams], dtype=np.float32)
        vec2 = np.array([counter2.get(ng, 0) for ng in all_ngrams], dtype=np.float32)

        # Calculer la similarité cosinus
        return self._cosine_similarity(vec1, vec2)

    def _semantic_similarity(self, s1: str, s2: str) -> float:
        """
        Calcule la similarité sémantique avec embeddings.

        Args:
            s1: Première chaîne
            s2: Deuxième chaîne

        Returns:
            Score de similarité sémantique [0, 1]
        """
        if not s1 or not s2:
            return 0.0

        # Récupérer les embeddings (avec cache)
        emb1 = self._get_embedding(s1)
        emb2 = self._get_embedding(s2)

        # Calculer la similarité cosinus
        cos_sim = self._cosine_similarity(emb1, emb2)

        # Normaliser entre 0 et 1
        return (cos_sim + 1) / 2

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Récupère l'embedding d'un texte via Ollama (avec cache).

        Args:
            text: Texte à encoder

        Returns:
            Vecteur d'embedding
        """
        # Vérifier le cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Appeler l'API Ollama pour obtenir l'embedding
        try:
            payload = {
                "model": self.embedding_model_name,
                "input": text
            }

            response = requests.post(
                self.embeddings_endpoint,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result.get("embedding", []), dtype=np.float32)

                # Mettre en cache
                self._embedding_cache[text] = embedding

                return embedding
            else:
                print(f"[!] Erreur API Ollama: {response.status_code}")
                # Retourner un vecteur vide en cas d'erreur
                return np.zeros(384, dtype=np.float32)

        except Exception as e:
            print(f"[!] Erreur lors de l'obtention de l'embedding: {e}")
            # Retourner un vecteur vide en cas d'erreur
            return np.zeros(384, dtype=np.float32)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.
        Utilise le GPU si disponible.

        Args:
            v1: Premier vecteur
            v2: Deuxième vecteur

        Returns:
            Similarité cosinus [-1, 1]
        """
        if self.use_gpu:
            # Conversion vers GPU
            v1_gpu = cp.asarray(v1)
            v2_gpu = cp.asarray(v2)

            dot_product = cp.dot(v1_gpu, v2_gpu)
            norm1 = cp.linalg.norm(v1_gpu)
            norm2 = cp.linalg.norm(v2_gpu)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            result = dot_product / (norm1 * norm2)
            return float(result.get())  # Convertir vers CPU
        else:
            # Calcul CPU standard
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

    @staticmethod
    def _normalize_string(s: str) -> str:
        """
        Normalise une chaîne pour le matching lexical (utilise cache LRU).

        Args:
            s: Chaîne à normaliser

        Returns:
            Chaîne normalisée
        """
        return _normalize_cached(s)

    def batch_similarity(self, lemma: str, ontology_terms: list) -> Dict[str, float]:
        """
        Calcule la similarité d'un lemme avec plusieurs termes ontologiques.
        Utilise le GPU pour le calcul batch si disponible.

        Args:
            lemma: Lemme à comparer
            ontology_terms: Liste de termes ontologiques

        Returns:
            Dictionnaire {terme: score}
        """
        if not ontology_terms:
            return {}

        # Pour l'algorithme sémantique ou hybride avec GPU, utiliser le batch GPU
        if self.use_gpu and self.algorithm in ("semantic", "hybrid"):
            return self._batch_similarity_gpu(lemma, ontology_terms)

        # Sinon, calcul séquentiel standard
        similarities = {}
        for term in ontology_terms:
            similarities[term] = self.calculate_similarity(lemma, term)
        return similarities

    def _batch_similarity_gpu(self, lemma: str, ontology_terms: List[str]) -> Dict[str, float]:
        """
        Calcul batch de similarités sémantiques avec GPU.
        Optimisé pour les grands ensembles de termes.

        Args:
            lemma: Lemme à comparer
            ontology_terms: Liste de termes ontologiques

        Returns:
            Dictionnaire {terme: score}
        """
        # Obtenir l'embedding du lemme
        lemma_emb = self._get_embedding(lemma)
        lemma_gpu = cp.asarray(lemma_emb)
        lemma_norm = cp.linalg.norm(lemma_gpu)

        if lemma_norm == 0:
            return {term: 0.0 for term in ontology_terms}

        # Construire la matrice d'embeddings pour tous les termes
        embeddings = []
        for term in ontology_terms:
            emb = self._get_embedding(term)
            embeddings.append(emb)

        # Convertir en matrice GPU
        embeddings_matrix = cp.asarray(np.array(embeddings, dtype=np.float32))

        # Calcul vectorisé des similarités cosinus
        # Normaliser chaque ligne (embedding de terme)
        norms = cp.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms = cp.where(norms == 0, 1.0, norms)  # Éviter division par zéro
        embeddings_normalized = embeddings_matrix / norms

        # Normaliser le lemme
        lemma_normalized = lemma_gpu / lemma_norm

        # Produit scalaire vectorisé (similarités cosinus)
        cos_similarities = cp.dot(embeddings_normalized, lemma_normalized)

        # Normaliser entre 0 et 1 (comme _semantic_similarity)
        normalized_sims = (cos_similarities + 1) / 2

        # Convertir vers CPU
        results = normalized_sims.get()

        # Si hybride, combiner avec similarité lexicale
        similarities = {}
        lemma_norm_str = self._normalize_string(lemma)

        for i, term in enumerate(ontology_terms):
            sem_score = float(results[i])

            if self.algorithm == "hybrid":
                term_norm = self._normalize_string(term)
                lex_score = self._lexical_similarity(lemma_norm_str, term_norm)
                similarities[term] = max(lex_score, sem_score)
            else:
                similarities[term] = sem_score

        return similarities

    def find_best_match(self, lemma: str, ontology_terms: list, threshold: float = 0.0) -> tuple:
        """
        Trouve le meilleur match pour un lemme.

        Args:
            lemma: Lemme à matcher
            ontology_terms: Liste de termes ontologiques
            threshold: Seuil minimum de similarité

        Returns:
            Tuple (meilleur_terme, score) ou (None, 0.0)
        """
        if not ontology_terms:
            return None, 0.0

        best_term = None
        best_score = 0.0

        for term in ontology_terms:
            score = self.calculate_similarity(lemma, term)
            if score > best_score and score >= threshold:
                best_score = score
                best_term = term

        return best_term, best_score

    def clear_cache(self):
        """Vide le cache des embeddings (CPU et GPU)."""
        self._embedding_cache.clear()
        if self.use_gpu and self._gpu_embedding_cache:
            self._gpu_embedding_cache.clear()
        print("[CACHE] Cache d'embeddings vide")

    def get_cache_size(self) -> int:
        """Retourne la taille du cache."""
        return len(self._embedding_cache)
