# -*- coding: utf-8 -*-
"""
Pushover Curve Bilinearization — Streamlit App
Méthode : Vy = Vmax, égalité des énergies (Eurocode 8)
Author  : Yacine (refactored for Streamlit Cloud deployment)
"""

import io
import numpy as np
import pandas as pd
import matplotlib

# NumPy 2.x removed np.trapz → use np.trapezoid, fall back for older versions
_trapz = getattr(np, "trapezoid", None) or np.trapz
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy import optimize, signal, interpolate

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pushover Bilinéarisation",
    page_icon="🏗️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# ENGINEERING FUNCTIONS  (unchanged from original script)
# ─────────────────────────────────────────────────────────────────────────────

def lire_donnees_excel(fichier):
    """
    Lit un fichier Excel uploadé via Streamlit.
    Retourne (deplacements, forces, forces_brutes) ou (None, None, None).
    """
    try:
        df = pd.read_excel(fichier, header=None)

        # Supprimer les lignes entièrement vides
        df.dropna(how="all", inplace=True)

        if df.shape[1] < 2:
            st.error("❌ Le fichier doit contenir au moins **2 colonnes** "
                     "(déplacement | force).")
            return None, None, None

        # Détecter si la première ligne est un en-tête textuel
        first_row = df.iloc[0]
        if not pd.api.types.is_numeric_dtype(type(first_row.iloc[0])):
            try:
                float(first_row.iloc[0])
            except (ValueError, TypeError):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)

        # Prendre les deux premières colonnes numériques
        col_dep = df.iloc[:, 0]
        col_for = df.iloc[:, 1]

        # Convertir en float, invalider les non-numériques
        col_dep = pd.to_numeric(col_dep, errors="coerce")
        col_for = pd.to_numeric(col_for, errors="coerce")

        n_nan = col_dep.isna().sum() + col_for.isna().sum()
        if n_nan > 0:
            st.warning(f"⚠️ {n_nan} valeur(s) non numérique(s) ignorée(s).")

        df_clean = pd.DataFrame({"d": col_dep, "F": col_for}).dropna()

        if len(df_clean) < 5:
            st.error("❌ Pas assez de points valides (minimum 5 requis).")
            return None, None, None

        if (df_clean["d"] < 0).any() or (df_clean["F"] < 0).any():
            st.warning("⚠️ Des valeurs négatives ont été détectées. "
                       "Vérifiez que le fichier représente bien une courbe pushover.")

        deplacements = df_clean["d"].values.astype(float)
        forces = df_clean["F"].values.astype(float)

        # Trier par déplacement croissant
        sort_idx = np.argsort(deplacements)
        deplacements = deplacements[sort_idx]
        forces = forces[sort_idx]

        return deplacements, forces, forces.copy()

    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
        return None, None, None


def lisser_courbe(deplacements, forces, methode="savgol", fenetre=None, degre=2):
    """Lissage de la courbe pushover (identique au script original)."""
    if methode == "aucun" or methode is None:
        return forces.copy()
    if len(deplacements) < 10:
        return forces.copy()

    if fenetre is None:
        fenetre = max(5, min(51, len(deplacements) // 10))
        if fenetre % 2 == 0:
            fenetre += 1

    try:
        if methode == "savgol":
            fenetre = min(fenetre, len(forces))
            if fenetre % 2 == 0:
                fenetre += 1
            forces_lissees = signal.savgol_filter(forces, fenetre, degre)
        elif methode == "moyenne_mobile":
            fenetre = min(fenetre, len(forces))
            forces_lissees = np.convolve(forces, np.ones(fenetre) / fenetre, mode="same")
        elif methode == "spline":
            tck = interpolate.splrep(deplacements, forces, s=len(forces) * 0.1)
            forces_lissees = interpolate.splev(deplacements, tck)
        else:
            return forces.copy()

        forces_lissees = np.maximum(forces_lissees, 0)
        return forces_lissees

    except Exception:
        return forces.copy()


def trouver_vmax_dmax(deplacements, forces):
    """Trouve le point de force maximale."""
    idx_max = np.argmax(forces)
    return forces[idx_max], deplacements[idx_max], idx_max


def calculer_aire_sous_la_courbe(deplacements, forces, d_limit=None):
    """Calcule l'aire sous la courbe pushover par intégration trapézoïdale."""
    if d_limit is None:
        d_limit = deplacements[-1]

    mask = deplacements <= d_limit
    if np.sum(mask) < 2:
        d_vals, f_vals = deplacements, forces
    else:
        d_vals, f_vals = deplacements[mask], forces[mask]

    if d_vals[-1] < d_limit:
        f_interp = np.interp(d_limit, deplacements, forces)
        d_vals = np.append(d_vals, d_limit)
        f_vals = np.append(f_vals, f_interp)

    return _trapz(f_vals, d_vals)


def estimer_rigidite_initiale(deplacements, forces):
    """Estime la rigidité initiale Ke par régression linéaire."""
    if len(deplacements) < 5:
        return max(
            (forces[1] - forces[0]) / (deplacements[1] - deplacements[0]), 1e-6
        )

    forces_np = np.array(forces)
    deplacements_np = np.array(deplacements)

    idx_max = np.argmax(forces_np)
    u_max = deplacements_np[idx_max]
    V30 = 0.3 * np.max(forces_np)

    condition = (forces_np <= V30) & (deplacements_np <= u_max)
    idx_30 = np.where(condition)[0]

    if len(idx_30) < 3:
        idx_30 = np.where(forces_np <= 0.5 * np.max(forces_np))[0]
        if len(idx_30) < 2:
            idx_30 = range(min(5, len(forces_np)))

    x = deplacements_np[idx_30]
    y = forces_np[idx_30]
    A = np.vstack([x, np.ones(len(x))]).T
    ke, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return max(ke, 1e-6)


def bilinearisation_ecrouissage_vy_egal_vmax(deplacements, forces, d_max=None, Vmax=None):
    """
    Bilinéarisation avec écrouissage par égalité d'énergie.
    Version : Vy = Vmax (identique au script original).
    """
    if d_max is None or Vmax is None:
        Vmax_calc, d_max_calc, _ = trouver_vmax_dmax(deplacements, forces)
        if d_max is None:
            d_max = d_max_calc
        if Vmax is None:
            Vmax = Vmax_calc

    Vy = Vmax
    aire_reelle = calculer_aire_sous_la_courbe(deplacements, forces, d_max)
    Ke_est = estimer_rigidite_initiale(deplacements, forces)

    # Formule analytique : dy = 2*(Vmax*d_max - Aire_reelle) / Vmax
    dy_initial = 2 * (Vmax * d_max - aire_reelle) / Vmax
    dy_initial = max(dy_initial, 0.01 * d_max)
    dy_initial = min(dy_initial, 0.70 * d_max)

    def erreur_energie(dy):
        if dy <= 0 or dy >= d_max:
            return 1e10
        Ke_loc = Vy / dy if dy > 0 else 0
        aire_triangle = 0.5 * Vy * dy
        aire_trapeze = 0.5 * (Vy + Vmax) * (d_max - dy)
        aire_bilin = aire_triangle + aire_trapeze
        erreur_rel = abs(aire_bilin - aire_reelle) / aire_reelle

        if dy < 0.05 * d_max:
            erreur_rel += 1.0
        if dy > 0.80 * d_max:
            erreur_rel += 1.0
        if Ke_est > 0:
            ratio_ke = Ke_loc / Ke_est
            if ratio_ke > 2.0 or ratio_ke < 0.3:
                erreur_rel += abs(ratio_ke - 1.0)
        return erreur_rel

    try:
        bounds = (0.01 * d_max, 0.80 * d_max)
        resultat = optimize.minimize_scalar(erreur_energie, bounds=bounds, method="bounded")

        if resultat.success:
            dy_opt = resultat.x
            Ke = Vy / dy_opt if dy_opt > 0 else 0
            Kp = 0.0
            aire_triangle = 0.5 * Vy * dy_opt
            aire_trapeze = 0.5 * (Vy + Vmax) * (d_max - dy_opt)
            aire_bilineaire = aire_triangle + aire_trapeze
            return Vy, dy_opt, Ke, Kp, d_max, Vmax, aire_bilineaire
        else:
            raise RuntimeError("Optimisation échouée")

    except Exception:
        Ke = Vy / dy_initial if dy_initial > 0 else 0
        Kp = 0.0
        aire_triangle = 0.5 * Vy * dy_initial
        aire_trapeze = 0.5 * (Vy + Vmax) * (d_max - dy_initial)
        aire_bilineaire = aire_triangle + aire_trapeze
        return Vy, dy_initial, Ke, Kp, d_max, Vmax, aire_bilineaire


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (unchanged appearance from original script)
# ─────────────────────────────────────────────────────────────────────────────

def creer_figure(deplacements, forces_brutes, forces_lissees, resultats, methode_lissage):
    """
    Crée la figure 2×2 identique au script original.
    Retourne un objet matplotlib Figure.
    """
    Vy      = resultats["Vy"]
    dy      = resultats["dy"]
    Ke      = resultats["Ke"]
    d_max   = resultats["d_max"]
    Vmax    = resultats["Vmax"]
    aire_reelle     = resultats["aire_reelle"]
    aire_bilineaire = resultats["aire_bilineaire"]
    diff_aire_pourcent = resultats["diff_aire_pourcent"]

    type_courbe = f"Courbe lissée ({methode_lissage})" if forces_lissees is not None else "Courbe brute"
    forces_plot = forces_lissees if forces_lissees is not None else forces_brutes

    d_bilin = np.array([0, dy, d_max])
    V_bilin = np.array([0, Vmax, Vmax])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Graphique 1 : Courbes comparées ────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(deplacements, forces_brutes, "b-", linewidth=1, alpha=0.4,
             label="Données brutes")
    if forces_lissees is not None:
        ax1.plot(deplacements, forces_lissees, "b-", linewidth=2.5,
                 label=type_courbe)
    else:
        ax1.plot(deplacements, forces_brutes, "b-", linewidth=2.5,
                 label="Courbe pushover")

    ax1.plot(d_bilin, V_bilin, "r--", linewidth=2,
             label="Courbe bilinéaire (Vy = Vmax)")
    ax1.plot(d_max, Vmax, "ro", markersize=10, label=f"Vmax = {Vmax:.1f}")
    ax1.plot(dy, Vmax,   "go", markersize=10, label=f"dy = {dy:.4f}")
    ax1.set_xlabel("Déplacement", fontsize=12)
    ax1.set_ylabel("Force de cisaillement", fontsize=12)
    ax1.set_title(f"Bilinéarisation - {type_courbe} (Vy = Vmax)", fontsize=14)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Graphique 2 : Zoom transition élastique-plastique ─────────────────
    ax2 = axes[0, 1]
    zoom_factor = 3.0
    x_min = max(0, dy - dy / zoom_factor)
    x_max = min(d_max, dy + dy / zoom_factor)

    mask_zoom = (deplacements >= x_min) & (deplacements <= x_max)
    if np.any(mask_zoom):
        ax2.plot(deplacements[mask_zoom], forces_plot[mask_zoom],
                 "b-", linewidth=2, label=type_courbe)
        ax2.plot(d_bilin, V_bilin, "r--", linewidth=2, label="Bilinéaire")
        ax2.plot(dy, Vmax, "go", markersize=8, label="dy")

        x_tang = np.array([x_min + (x_max - x_min) * 0.3,
                            x_min + (x_max - x_min) * 0.7])
        y_tang = Ke * x_tang
        ax2.plot(x_tang, y_tang, "g:", linewidth=1.5, label=f"Ke = {Ke:.1f}")

    ax2.set_xlabel("Déplacement", fontsize=12)
    ax2.set_ylabel("Force de cisaillement", fontsize=12)
    ax2.set_title("Zoom: Transition élastique-plastique", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Graphique 3 : Comparaison des aires ────────────────────────────────
    ax3 = axes[1, 0]
    categories = ["Aire réelle", "Aire bilinéaire"]
    valeurs    = [aire_reelle, aire_bilineaire]
    couleurs   = ["blue", "red"]
    bars = ax3.bar(categories, valeurs, color=couleurs, alpha=0.7, width=0.6)
    ax3.set_ylabel("Aire (énergie)", fontsize=12)
    ax3.set_title("Égalité des aires/énergies", fontsize=12)
    ax3.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, valeurs):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=11)

    color_diff = ("green"  if diff_aire_pourcent < 5
                  else "orange" if diff_aire_pourcent < 10
                  else "red")
    ax3.text(0.5, 0.9,
             f"Différence: {diff_aire_pourcent:.2f}%",
             ha="center", fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor=color_diff, alpha=0.3),
             transform=ax3.transAxes)

    # ── Graphique 4 : Courbe pushover avec zones remplies ──────────────────
    ax4 = axes[1, 1]
    ax4.plot(deplacements, forces_plot, "b-", linewidth=2, label="Courbe pushover")
    ax4.fill_between(deplacements, forces_plot, 0, alpha=0.2, color="blue")

    d_fill = np.array([0, dy, d_max, d_max, dy, 0])
    V_fill = np.array([0, Vmax, Vmax, 0, 0, 0])
    ax4.fill(d_fill, V_fill, "red", alpha=0.15, label="Aire bilinéaire")
    ax4.plot(d_bilin, V_bilin, "r--", linewidth=2, label="Bilinéaire")
    ax4.plot(d_max, Vmax, "ro", markersize=10, label=f"Vmax = {Vmax:.1f}")
    ax4.axvline(x=dy, color="g", linestyle=":", linewidth=1.5,
                label=f"dy = {dy:.4f}")
    ax4.set_xlabel("Déplacement", fontsize=12)
    ax4.set_ylabel("Force de cisaillement", fontsize=12)
    ax4.set_title("Courbe pushover et bilinéarisation (Vy = Vmax)", fontsize=12)
    ax4.legend(loc="best", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def analyser_pushover_depuis_donnees(deplacements, forces_brutes,
                                     lissage=True, methode_lissage="savgol"):
    """
    Reproduit exactement le pipeline de analyser_pushover() du script original,
    mais en acceptant des arrays déjà chargés (depuis Excel).
    """
    # Lissage
    if lissage and methode_lissage != "aucun":
        forces = lisser_courbe(deplacements, forces_brutes, methode=methode_lissage)
        forces_lissees = forces
    else:
        forces = forces_brutes.copy()
        forces_lissees = None

    # Bilinéarisation
    Vy, dy, Ke, Kp, d_max, Vmax, aire_bilineaire = \
        bilinearisation_ecrouissage_vy_egal_vmax(deplacements, forces)

    # Calculs complémentaires
    aire_reelle = calculer_aire_sous_la_courbe(deplacements, forces, d_max)
    diff_aire_pourcent = abs(aire_bilineaire - aire_reelle) / aire_reelle * 100
    ductilite = d_max / dy if dy > 0 else 0

    resultats = {
        "deplacements":    deplacements,
        "forces_brutes":   forces_brutes,
        "forces_lissees":  forces_lissees,
        "Vmax":            Vmax,
        "d_max":           d_max,
        "Vy":              Vy,
        "dy":              dy,
        "Ke":              Ke,
        "Kp":              Kp,
        "ratio_ecrouissage": 0.0,
        "ductilite":       ductilite,
        "sur_resistance":  1.0,
        "aire_reelle":     aire_reelle,
        "aire_bilineaire": aire_bilineaire,
        "diff_aire_pourcent": diff_aire_pourcent,
        "methode_lissage": methode_lissage if lissage else "aucun",
    }
    return resultats, forces_lissees


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── En-tête ──────────────────────────────────────────────────────────────
    st.title("🏗️ Bilinéarisation de courbe Pushover")
    st.markdown(
        "**Méthode** : Vy = Vmax · Égalité des énergies · "
        "Écrouissage nul — *conforme Eurocode 8*"
    )
    st.divider()

    # ── Sidebar : paramètres ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Paramètres d'analyse")

        lissage = st.toggle("Appliquer un lissage", value=True)

        methode_lissage = "aucun"
        if lissage:
            methode_lissage = st.selectbox(
                "Méthode de lissage",
                options=["savgol", "moyenne_mobile", "spline"],
                format_func=lambda x: {
                    "savgol":          "Savitzky-Golay (recommandé)",
                    "moyenne_mobile":  "Moyenne mobile",
                    "spline":          "Spline cubique",
                }[x],
            )

        st.divider()
        st.markdown(
            "**Format Excel attendu :**\n"
            "- Colonne 1 : Déplacement (mm)\n"
            "- Colonne 2 : Force (kN)\n"
            "- En-tête optionnel"
        )
        st.divider()
        st.caption("Développé par Yacine · Streamlit Cloud")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "📂 Charger la courbe pushover (fichier Excel)",
        type=["xlsx", "xls"],
        help="Deux colonnes : Déplacement | Force. L'en-tête est optionnel.",
    )

    if uploaded is None:
        st.info("⬆️ Veuillez charger un fichier Excel pour démarrer l'analyse.")
        # Exemple de structure attendue
        with st.expander("📋 Exemple de structure du fichier Excel"):
            demo = pd.DataFrame({
                "Déplacement (mm)": [0, 10, 25, 50, 80, 120, 160, 200, 250],
                "Force (kN)":       [0, 300, 700, 1100, 1400, 1520, 1524, 1500, 1250],
            })
            st.dataframe(demo, use_container_width=True)
        return

    # ── Lecture des données ───────────────────────────────────────────────────
    deplacements, forces_brutes, _ = lire_donnees_excel(uploaded)
    if deplacements is None:
        return

    st.success(
        f"✅ Fichier lu avec succès · **{len(deplacements)}** points · "
        f"d_max = {deplacements[-1]:.4f} · F_max = {np.max(forces_brutes):.2f}"
    )

    # Aperçu des données
    with st.expander("🔍 Aperçu des données importées"):
        df_preview = pd.DataFrame({
            "Déplacement": deplacements,
            "Force":       forces_brutes,
        })
        st.dataframe(df_preview, use_container_width=True, height=250)

    # ── Calculs ───────────────────────────────────────────────────────────────
    with st.spinner("⏳ Bilinéarisation en cours…"):
        resultats, forces_lissees = analyser_pushover_depuis_donnees(
            deplacements, forces_brutes,
            lissage=lissage,
            methode_lissage=methode_lissage,
        )

    # ── Métriques clés ────────────────────────────────────────────────────────
    st.subheader("📊 Résultats")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Vmax = Vy",         f"{resultats['Vmax']:.2f} kN")
    col2.metric("d_max",             f"{resultats['d_max']:.4f} mm")
    col3.metric("dy",                f"{resultats['dy']:.4f} mm")
    col4.metric("Ke",                f"{resultats['Ke']:.1f} kN/mm")
    col5.metric("Ductilité μ",       f"{resultats['ductilite']:.3f}")

    col6, col7, col8 = st.columns(3)
    col6.metric("Aire réelle",       f"{resultats['aire_reelle']:.1f}")
    col7.metric("Aire bilinéaire",   f"{resultats['aire_bilineaire']:.1f}")
    diff = resultats["diff_aire_pourcent"]
    col8.metric("Différence d'aire", f"{diff:.2f}%",
                delta=None,
                help="< 5 % : excellent · < 10 % : acceptable")

    st.divider()

    # ── Figure 2×2 ────────────────────────────────────────────────────────────
    st.subheader("📈 Graphiques de bilinéarisation")

    fig = creer_figure(
        deplacements, forces_brutes, forces_lissees,
        resultats, methode_lissage
    )
    st.pyplot(fig, use_container_width=True)

    # ── Export PNG ────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Télécharger la figure (PNG)",
        data=buf,
        file_name="bilinearisation_pushover.png",
        mime="image/png",
    )

    # ── Export CSV des résultats ───────────────────────────────────────────────
    df_resultats = pd.DataFrame([{
        "Vmax (kN)":             resultats["Vmax"],
        "d_max (mm)":            resultats["d_max"],
        "Vy (kN)":               resultats["Vy"],
        "dy (mm)":               resultats["dy"],
        "Ke (kN/mm)":            resultats["Ke"],
        "Kp (kN/mm)":            resultats["Kp"],
        "Ductilité μ":           resultats["ductilite"],
        "Sur-résistance Ω":      resultats["sur_resistance"],
        "Aire réelle":           resultats["aire_reelle"],
        "Aire bilinéaire":       resultats["aire_bilineaire"],
        "Différence aire (%)":   resultats["diff_aire_pourcent"],
        "Méthode lissage":       resultats["methode_lissage"],
    }])

    csv_buf = io.StringIO()
    df_resultats.to_csv(csv_buf, index=False, sep=";", decimal=",")
    st.download_button(
        label="⬇️ Télécharger les résultats (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="resultats_bilinearisation.csv",
        mime="text/csv",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
