# Bilinéarisation de Courbe Pushover - Eurocode 8 (Vy = Vmax)

Cette application web interactive permet de réaliser la bilinéarisation d'une courbe de pushover (force-déplacement) selon les dispositions de l'Eurocode 8, en utilisant l'approche d'équivalence d'énergie sous la contrainte $V_y = V_{max}$.

## Fonctionnalités

*   **Import Excel direct** : Chargement de fichiers Excel (`.xlsx`, `.xls`) contenant les colonnes de déplacement (mm) et de force (kN).
*   **Validation robuste** : Détection automatique des en-têtes et gestion des valeurs manquantes.
*   **Lissage de signal** : Intégration des méthodes de lissage (Savitzky-Golay, moyenne mobile, spline).
*   **Calculs techniques** : Évaluation automatique de $V_{max}$, $d_{max}$, $V_y$, $d_y$, de la rigidité initiale $K_e$ et de la ductilité $\mu$.
*   **Visualisation conforme** : Génération d'un rapport graphique 2x2 identique à la version originale.

## Installation locale

1. Cloner le dépôt ou télécharger les fichiers :
   ```bash
   git clone https://github.com/votre-utilisateur/votre-depot.git
   cd votre-depot
