# 🏗️ Pushover Bilinéarisation — Streamlit App

Analyse et bilinéarisation de courbes pushover selon **Eurocode 8**.  
Méthode : **Vy = Vmax**, égalité des énergies, écrouissage nul.

---

## Aperçu

L'application produit une figure 2×2 comprenant :

| Graphique | Contenu |
|-----------|---------|
| ↖ Haut gauche | Courbe brute + lissée + bilinéaire |
| ↗ Haut droite | Zoom sur la transition élastique-plastique |
| ↙ Bas gauche  | Comparaison des aires (égalité des énergies) |
| ↘ Bas droite  | Courbe pushover avec zones remplies |

---

## Structure du projet

```
pushover_app/
├── app.py              # Application Streamlit principale
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

---

## Format du fichier Excel d'entrée

Le fichier Excel doit contenir **exactement deux colonnes** :

| Colonne 1            | Colonne 2   |
|----------------------|-------------|
| Déplacement (mm)     | Force (kN)  |

- L'en-tête est **optionnel** : l'app le détecte automatiquement.
- Les valeurs non numériques sont ignorées avec un avertissement.
- Minimum **5 points** valides requis.

Exemple :

```
Déplacement    Force
0              0
10             320
50             1100
120            1510
140            1524
200            1490
270            1230
```

---

## Lancer l'app en local

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-compte>/pushover-bilinearisation.git
cd pushover-bilinearisation

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Streamlit
streamlit run app.py
```

L'app s'ouvre automatiquement dans votre navigateur à `http://localhost:8501`.

---

## Déployer sur Streamlit Cloud

1. **Pousser le projet sur GitHub** :
   ```bash
   git init
   git add .
   git commit -m "Initial commit — pushover bilinearisation app"
   git remote add origin https://github.com/<votre-compte>/pushover-bilinearisation.git
   git push -u origin main
   ```

2. **Créer un compte** sur [streamlit.io/cloud](https://streamlit.io/cloud) (gratuit).

3. **New app** → choisir le dépôt GitHub → branche `main` → fichier principal `app.py`.

4. Cliquer **Deploy** — Streamlit Cloud installe automatiquement les dépendances via `requirements.txt`.

> ℹ️ Aucune modification de code n'est nécessaire entre l'exécution locale et le déploiement Cloud.

---

## Paramètres disponibles (sidebar)

| Paramètre        | Options                                      |
|------------------|----------------------------------------------|
| Lissage          | Activé / Désactivé                          |
| Méthode lissage  | Savitzky-Golay · Moyenne mobile · Spline    |

---

## Résultats exportables

- **PNG** : figure 2×2 haute résolution (150 dpi)  
- **CSV** : tableau des résultats numériques (séparateur `;`, décimale `,`)

---

## Dépendances

| Bibliothèque | Rôle |
|--------------|------|
| `streamlit`  | Interface web |
| `numpy`      | Calculs numériques |
| `pandas`     | Lecture Excel |
| `matplotlib` | Graphiques |
| `scipy`      | Lissage + optimisation |
| `openpyxl`   | Lecture `.xlsx` |
| `xlrd`       | Lecture `.xls` |

---

## Auteur

Développé par **Yacine** — méthode de bilinéarisation conforme Eurocode 8.
