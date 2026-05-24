---

### 3. `app.py`

```python
# -*- coding: utf-8 -*-
"""
Application de Bilinéarisation Pushover (Eurocode 8) - Vy = Vmax
Développé avec Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, signal, interpolate
import io

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Bilinéarisation Pushover - Eurocode 8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style global de l'application
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1e3d59;
    }
    h3 {
        color: #17b978;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS DE CALCUL ET DE TRAITEMENT (Conservation de l'algorithme original)
# ============================================================================

def lisser_courbe(deplacements, forces, methode='savgol', fenetre=None, degre=2):
    """
    Applique un lissage à la courbe pushover.
    """
    if methode == 'aucun' or methode is None:
        return forces.copy()
    
    if len(deplacements) < 10:
        return forces.copy()
    
    if fenetre is None:
        fenetre = max(5, min(51, len(deplacements) // 10))
        if fenetre % 2 == 0:
            fenetre += 1
    
    try:
        if methode == 'savgol':
            fenetre = min(fenetre, len(forces))
            if fenetre % 2 == 0:
                fenetre += 1
            forces_lissees = signal.savgol_filter(forces, fenetre, degre)
            
        elif methode == 'moyenne_mobile':
            fenetre = min(fenetre, len(forces))
            forces_lissees = np.convolve(forces, np.ones(fenetre)/fenetre, mode='same')
                
        elif methode == 'spline':
            tck = interpolate.splrep(deplacements, forces, s=len(forces)*0.1)
            forces_lissees = interpolate.splev(deplacements, tck)
            
        else:
            return forces.copy()
        
        forces_lissees = np.maximum(forces_lissees, 0)
        return forces_lissees
        
    except Exception as e:
        st.warning(f"Erreur de lissage: {e}. Utilisation de la courbe brute.")
        return forces.copy()

def trouver_vmax_dmax(deplacements, forces):
    idx_max = np.argmax(forces)
    Vmax = forces[idx_max]
    d_max = deplacements[idx_max]
    return Vmax, d_max, idx_max

def calculer_aire_sous_la_courbe(deplacements, forces, d_limit=None):
    if d_limit is None:
        d_limit = deplacements[-1]
    
    mask = deplacements <= d_limit
    
    if np.sum(mask) < 2:
        d_vals = deplacements
        f_vals = forces
    else:
        d_vals = deplacements[mask]
        f_vals = forces[mask]
    
    if d_vals[-1] < d_limit:
        f_interp = np.interp(d_limit, deplacements, forces)
        d_vals = np.append(d_vals, d_limit)
        f_vals = np.append(f_vals, f_interp)
    
    return np.trapz(f_vals, d_vals)

def estimer_rigidite_initiale(deplacements, forces):
    if len(deplacements) < 5:
        return max((forces[1] - forces[0]) / (deplacements[1] - deplacements[0]), 1e-6)
    
    forces_np = np.array(forces)
    deplacements_np = np.array(deplacements)
    
    idx_max = np.argmax(forces_np)
    u_max = deplacements_np[idx_max]
    
    V30 = 0.3 * np.max(forces_np)
    condition = (forces_np <= V30) & (deplacements_np <= u_max)
    idx_30 = np.where(condition)[0]
    
    if len(idx_30) < 3:
        idx_30 = np.where(forces <= 0.5 * np.max(forces))[0]
        if len(idx_30) < 2:
            idx_30 = range(min(5, len(forces)))
    
    x = deplacements[idx_30]
    y = forces[idx_30]
    
    A = np.vstack([x, np.ones(len(x))]).T
    ke, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return max(ke, 1e-6)

def bilinearisation_ecrouissage_vy_egal_vmax(deplacements, forces, d_max=None, Vmax=None):
    if d_max is None or Vmax is None:
        Vmax_calc, d_max_calc, _ = trouver_vmax_dmax(deplacements, forces)
        if d_max is None:
            d_max = d_max_calc
        if Vmax is None:
            Vmax = Vmax_calc
    
    Vy = Vmax
    aire_reelle = calculer_aire_sous_la_courbe(deplacements, forces, d_max)
    Ke_est = estimer_rigidite_initiale(deplacements, forces)
    
    dy_initial = 2 * (Vmax * d_max - aire_reelle) / Vmax
    dy_initial = max(dy_initial, 0.01 * d_max)
    dy_initial = min(dy_initial, 0.7 * d_max)
    
    def erreur_energie(dy):
        if dy <= 0 or dy >= d_max:
            return 1e10
        
        Ke = Vy / dy if dy > 0 else 0
        aire_triangle = 0.5 * Vy * dy
        aire_trapeze = 0.5 * (Vy + Vmax) * (d_max - dy)
        aire_bilineaire = aire_triangle + aire_trapeze
        
        erreur_rel = abs(aire_bilineaire - aire_reelle) / aire_reelle
        
        if dy < 0.05 * d_max:
            erreur_rel += 1.0
        if dy > 0.8 * d_max:
            erreur_rel += 1.0
        
        if Ke_est > 0:
            ratio_ke = Ke / Ke_est
            if ratio_ke > 2.0 or ratio_ke < 0.3:
                erreur_rel += abs(ratio_ke - 1.0)
        
        return erreur_rel
    
    try:
        bounds = [(0.01 * d_max, 0.8 * d_max)]
        resultat = optimize.minimize_scalar(
            erreur_energie,
            bounds=bounds,
            method='bounded'
        )
        
        if resultat.success:
            dy_opt = resultat.x
            Ke = Vy / dy_opt if dy_opt > 0 else 0
            Kp = 0
            aire_triangle = 0.5 * Vy * dy_opt
            aire_trapeze = 0.5 * (Vy + Vmax) * (d_max - dy_opt)
            aire_bilineaire = aire_triangle + aire_trapeze
            return Vy, dy_opt, Ke, Kp, d_max, Vmax, aire_bilineaire
        else:
            raise RuntimeError("Optimisation échouée")
            
    except Exception:
        Ke = Vy / dy_initial if dy_initial > 0 else 0
        Kp = 0
        aire_triangle = 0.5 * Vy * dy_initial
        aire_trapeze = 0.5 * (Vy + Vmax) * (d_max - dy_initial)
        aire_bilineaire = aire_triangle + aire_trapeze
        return Vy, dy_initial, Ke, Kp, d_max, Vmax, aire_bilineaire

# ============================================================================
# LECTURE DE FICHIER ET GENERATION DE DONNEES DEMO
# ============================================================================

def charger_donnees_excel(uploaded_file):
    """
    Lit et valide les données du fichier Excel importé.
    Assure l'extraction automatique sans ou avec en-têtes.
    """
    try:
        df_raw = pd.read_excel(uploaded_file, header=None)
        if df_raw.shape[1] < 2:
            return None, "Le fichier Excel doit contenir au moins 2 colonnes (Déplacement et Force)."
        
        # Test de la présence d'en-têtes sur la première ligne
        premiere_ligne = df_raw.iloc[0, :2]
        is_header = False
        try:
            float(premiere_ligne.iloc[0])
            float(premiere_ligne.iloc[1])
        except (ValueError, TypeError):
            is_header = True
            
        if is_header:
            df = pd.read_excel(uploaded_file)
        else:
            df = df_raw.copy()
            df.columns = ["disp", "force"]
            
        # Conversion numérique et nettoyage
        col_disp = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        col_force = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        df_clean = pd.DataFrame({"disp": col_disp, "force": col_force}).dropna()
        
        if df_clean.empty:
            return None, "Aucune donnée numérique valide n'a été détectée dans les deux premières colonnes."
        
        if len(df_clean) < 10:
            return None, "Données insuffisantes. Le fichier doit comporter au moins 10 points de mesure."
            
        deplacements = df_clean["disp"].to_numpy()
        forces = df_clean["force"].to_numpy()
        
        # Tri
        idx_trie = np.argsort(deplacements)
        return deplacements[idx_trie], forces[idx_trie], None
        
    except Exception as e:
        return None, f"Erreur lors du traitement du fichier Excel : {str(e)}"

def generer_donnees_exemple():
    """
    Génère une courbe pushover réaliste proche du cas d'étude réel pour la démonstration.
    """
    d = np.linspace(0, 280, 150)
    forces_propres = []
    for x in d:
        if x <= 135:
            val = 1523.6 * (1.0 - (1.0 - x/135.0)**1.8)
        else:
            val = 1523.6 - (1523.6 - 1230.0) * ((x - 135.0) / (280.0 - 135.0))**1.2
        forces_propres.append(val)
        
    forces_propres = np.array(forces_propres)
    np.random.seed(42)
    bruit = np.random.normal(0, 15, len(d)) * (d / 280.0)**0.5
    forces_brutes = np.maximum(forces_propres + bruit, 0)
    forces_brutes[0] = 0.0
    return d, forces_brutes

# ============================================================================
# INTERFACE UTILISATEUR STREAMLIT (UI)
# ============================================================================

st.title("📈 Bilinéarisation Pushover (Eurocode 8)")
st.write("Outil d'analyse structurelle pour la bilinéarisation de courbes de capacité selon les critères énergétiques de l'Eurocode 8.")

# Barre latérale (Contrôles et imports)
st.sidebar.header("📁 Données d'entrée")
source_donnees = st.sidebar.radio(
    "Sélectionner la source :",
    ["Données de démonstration", "Importer un fichier Excel (.xlsx, .xls)"]
)

deplacements, forces_brutes = None, None
erreur_chargement = None

if source_donnees == "Importer un fichier Excel (.xlsx, .xls)":
    fichier_charge = st.sidebar.file_uploader(
        "Sélectionnez le fichier Excel (Col 1: Déplacement [mm], Col 2: Force [kN])",
        type=["xlsx", "xls"]
    )
    if fichier_charge is not None:
        deplacements, forces_brutes, erreur_chargement = charger_donnees_excel(fichier_charge)
        if erreur_chargement:
            st.sidebar.error(erreur_chargement)
    else:
        st.sidebar.info("Veuillez importer un fichier pour lancer l'analyse.")
else:
    deplacements, forces_brutes = generer_donnees_exemple()
    st.sidebar.success("Données de démonstration chargées.")

# Paramètres de lissage dans la barre latérale
if deplacements is not None and forces_brutes is not None:
    st.sidebar.header("⚙️ Paramètres de lissage")
    activer_lissage = st.sidebar.checkbox("Activer le lissage de la courbe", value=True)
    
    methode_lissage = 'aucun'
    fenetre_savgol = 15
    degre_savgol = 2
    
    if activer_lissage:
        methode_lissage = st.sidebar.selectbox(
            "Méthode de lissage",
            ["savgol", "moyenne_mobile", "spline"],
            index=0
        )
        
        auto_fenetre = st.sidebar.checkbox("Ajustement automatique de la fenêtre", value=True)
        if not auto_fenetre:
            fenetre_savgol = st.sidebar.slider(
                "Taille de la fenêtre",
                min_value=5,
                max_value=min(101, len(deplacements)),
                value=min(25, len(deplacements)//4),
                step=2
            )
        else:
            fenetre_savgol = None
            
        if methode_lissage == 'savgol':
            degre_savgol = st.sidebar.slider("Degré du polynôme", min_value=1, max_value=5, value=2)
    else:
        methode_lissage = 'aucun'

    # Traitement des données et bilinéarisation
    if activer_lissage and methode_lissage != 'aucun':
        forces_calculees = lisser_courbe(
            deplacements, forces_brutes, 
            methode=methode_lissage, 
            fenetre=fenetre_savgol, 
            degre=degre_savgol
        )
        type_courbe = f"Courbe lissée ({methode_lissage})"
    else:
        forces_calculees = forces_brutes.copy()
        type_courbe = "Courbe brute"

    # Algorithme de bilinéarisation
    Vy, dy, Ke, Kp, d_max, Vmax, aire_bilineaire = bilinearisation_ecrouissage_vy_egal_vmax(
        deplacements, forces_calculees
    )
    
    # Calculs complémentaires
    aire_reelle = calculer_aire_sous_la_courbe(deplacements, forces_calculees, d_max)
    diff_aire_pourcent = abs(aire_bilineaire - aire_reelle) / aire_reelle * 100
    ductilite = d_max / dy if dy > 0 else 1.0

    # ============================================================================
    # SECTION 1 : METRICS / RÉSULTATS PRINCIPAUX
    # ============================================================================
    st.subheader("📊 Résultats de la Bilinéarisation")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Force Max Vmax (kN)", value=f"{Vmax:.1f}")
    with col2:
        st.metric(label="Dépl. Max d_max (mm)", value=f"{d_max:.2f}")
    with col3:
        st.metric(label="Force d'écoulement Vy (kN)", value=f"{Vy:.1f}")
    with col4:
        st.metric(label="Dépl. d'écoulement dy (mm)", value=f"{dy:.4f}")
    with col5:
        st.metric(label="Ductilité globale μ", value=f"{ductilite:.3f}")

    col6, col7, col8, col9 = st.columns(4)
    with col6:
        st.metric(label="Rigidité élastique Ke (kN/mm)", value=f"{Ke:.2f}")
    with col7:
        st.metric(label="Aire Réelle", value=f"{aire_reelle:.1f}")
    with col8:
        st.metric(label="Aire Bilinéaire", value=f"{aire_bilineaire:.1f}")
    with col9:
        st.metric(label="Écart d'Énergie", value=f"{diff_aire_pourcent:.2f} %")

    # ============================================================================
    # SECTION 2 : FIGURES MATPLOTLIB (2X2 - Préservation absolue du style d'origine)
    # ============================================================================
    st.write("---")
    st.subheader("🎨 Diagrammes de Comportement")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # Graphique 1: Courbes comparées
    ax1 = axes[0, 0]
    ax1.plot(deplacements, forces_brutes, 'b-', linewidth=1, alpha=0.4, label='Données brutes')
    
    if activer_lissage and methode_lissage != 'aucun':
        ax1.plot(deplacements, forces_calculees, 'b-', linewidth=2.5, label=type_courbe)
    else:
        ax1.plot(deplacements, forces_calculees, 'b-', linewidth=2.5, label='Courbe pushover')
    
    d_bilin = np.array([0, dy, d_max])
    V_bilin = np.array([0, Vmax, Vmax])
    ax1.plot(d_bilin, V_bilin, 'r--', linewidth=2, label='Courbe bilinéaire (Vy = Vmax)')
    
    ax1.plot(d_max, Vmax, 'ro', markersize=10, label=f'Vmax = {Vmax:.1f}')
    ax1.plot(dy, Vmax, 'go', markersize=10, label=f'dy = {dy:.4f}')
    
    ax1.set_xlabel('Déplacement', fontsize=12)
    ax1.set_ylabel('Force de cisaillement', fontsize=12)
    ax1.set_title(f'Bilinéarisation - {type_courbe} (Vy = Vmax)', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Zoom sur la transition
    ax2 = axes[0, 1]
    zoom_factor = 3.0
    x_min = max(0, dy - dy/zoom_factor)
    x_max = min(d_max, dy + dy/zoom_factor)
    
    mask_zoom = (deplacements >= x_min) & (deplacements <= x_max)
    if np.any(mask_zoom):
        if activer_lissage and methode_lissage != 'aucun':
            ax2.plot(deplacements[mask_zoom], forces_calculees[mask_zoom], 'b-', linewidth=2, label=type_courbe)
        else:
            ax2.plot(deplacements[mask_zoom], forces_calculees[mask_zoom], 'b-', linewidth=2, label='Courbe')
        
        ax2.plot(d_bilin, V_bilin, 'r--', linewidth=2, label='Bilinéaire')
        ax2.plot(dy, Vmax, 'go', markersize=8, label='dy')
        
        x_tang = np.array([x_min + (x_max-x_min)*0.3, x_min + (x_max-x_min)*0.7])
        y_tang_elastique = Ke * x_tang
        ax2.plot(x_tang, y_tang_elastique, 'g:', linewidth=1.5, label=f'Ke = {Ke:.1f}')
        
        ax2.set_xlabel('Déplacement', fontsize=12)
        ax2.set_ylabel('Force de cisaillement', fontsize=12)
        ax2.set_title('Zoom: Transition élastique-plastique', fontsize=12)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # Graphique 3: Comparaison des aires
    ax3 = axes[1, 0]
    categories = ['Aire réelle', 'Aire bilinéaire']
    valeurs = [aire_reelle, aire_bilineaire]
    couleurs = ['blue', 'red']
    
    bars = ax3.bar(categories, valeurs, color=couleurs, alpha=0.7, width=0.6)
    ax3.set_ylabel('Aire (énergie)', fontsize=12)
    ax3.set_title('Égalité des aires/énergies', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, valeurs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}', ha='center', va='bottom', fontsize=11)
    
    color_box = 'green' if diff_aire_pourcent < 5 else 'orange' if diff_aire_pourcent < 10 else 'red'
    ax3.text(0.5, max(valeurs)*0.9, f'Différence: {diff_aire_pourcent:.2f}%',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color_box, alpha=0.3))
    
    # Graphique 4: Courbe pushover complète avec points caractéristiques
    ax4 = axes[1, 1]
    ax4.plot(deplacements, forces_calculees, 'b-', linewidth=2, label='Courbe pushover')
    ax4.fill_between(deplacements, forces_calculees, 0, alpha=0.2, color='blue')
    
    d_fill = np.array([0, dy, d_max, d_max, dy, 0])
    V_fill = np.array([0, Vmax, Vmax, 0, 0, 0])
    ax4.fill(d_fill, V_fill, 'red', alpha=0.15, label='Aire bilinéaire')
    
    ax4.plot(d_bilin, V_bilin, 'r--', linewidth=2, label='Bilinéaire')
    ax4.plot(d_max, Vmax, 'ro', markersize=10, label=f'Vmax = {Vmax:.1f}')
    ax4.axvline(x=dy, color='g', linestyle=':', linewidth=1.5, label=f'dy = {dy:.4f}')
    
    ax4.set_xlabel('Déplacement', fontsize=12)
    ax4.set_ylabel('Force de cisaillement', fontsize=12)
    ax4.set_title('Courbe pushover et bilinéarisation (Vy = Vmax)', fontsize=12)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ============================================================================
    # SECTION 3 : AFFICHAGE DU TABLEAU DE DONNÉES ET TELECHARGEMENT
    # ============================================================================
    st.write("---")
    st.subheader("📋 Tableau des Données")
    
    df_visualisation = pd.DataFrame({
        "Déplacement (mm)": deplacements,
        "Force brute (kN)": forces_brutes,
        "Force lissée (kN)": forces_calculees
    })
    
    col_table, col_download = st.columns([2, 1])
    
    with col_table:
        st.dataframe(df_visualisation.style.format("{:.3f}"), height=250)
        
    with col_download:
        st.write("### Extraire les données")
        
        # Préparation du fichier CSV de sortie des résultats
        results_txt = f"""------------------------------------------------------------
ANALYSE PUSHOVER - BILINEARISATION EUROCODE 8 (Vy = Vmax)
------------------------------------------------------------
Vmax (Force maximale)            : {Vmax:.4f} kN
d_max (Déplacement à Vmax)       : {d_max:.4f} mm
Vy (Force d'écoulement plastique): {Vy:.4f} kN
dy (Déplacement d'écoulement)    : {dy:.4f} mm
Ke (Rigidité élastique initiale) : {Ke:.4f} kN/mm
Kp (Rigidité plastique)          : {Kp:.4f} kN/mm
Ductilité globale (u = d_max/dy) : {ductilite:.4f}
Aire réelle (sous la courbe)     : {aire_reelle:.4f}
Aire bilinéaire (modélisée)      : {aire_bilineaire:.4f}
Écart d'énergie (%)              : {diff_aire_pourcent:.4f}%
Méthode de lissage appliquée     : {methode_lissage}
------------------------------------------------------------
"""
        st.download_button(
            label="💾 Télécharger le rapport technique (.txt)",
            data=results_txt,
            file_name="rapport_bilinearisation_pushover.txt",
            mime="text/plain"
        )
        
        # Préparation du fichier Excel des courbes lissées
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
            df_visualisation.to_excel(writer, index=False, sheet_name="Courbes")
        
        st.download_button(
            label="📊 Télécharger la courbe traitée (Excel)",
            data=buffer_excel.getvalue(),
            file_name="courbe_pushover_traitee.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
