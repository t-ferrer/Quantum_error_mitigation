"""
mirror_circuit_diagrams.py

Génère des schémas de circuits miroir Pauli-randomisés pour une présentation Canva.

Run:
    python visuals/mirror_circuit_diagrams.py

Sorties (dans visuals/) :
    1. mirror_schematic.png       — bloc-diagramme de la structure complète
    2. echo_vs_mirror.png         — comparaison annotée écho naïf vs miroir
    3. circuit_echo_qiskit.png    — circuit écho naïf 2-qubits (Qiskit)
    4. circuit_mirror_qiskit.png  — circuit miroir 2-qubits (Qiskit)
    5. success_prob_decay.png     — P(s) vs profondeur : écho ≡ 1 vs miroir qui décroît
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 250

# ─── Palette de couleurs ─────────────────────────────────────────────────────
C = dict(
    fwd    = '#2E86AB',   # bleu acier — Clifford avant
    inv    = '#3D9970',   # vert — quasi-inverse
    pdress = '#FFBF69',   # ambre clair — couches Pauli-dressed
    pmid   = '#FF9F1C',   # ambre — Paulis centraux
    meas   = '#845EC2',   # violet — mesure
    bad    = '#E84855',   # rouge — naïf/mauvais
    bg     = '#FFFFFF',
    text   = '#1A1A2E',
    wire   = '#BBBBBB',
    sep    = '#DDDDDD',
    edge   = '#444444',
)


def rbox(ax, x, y, w, h, fc, label, sub=None, tc='white', fs=12, a=1.0, lw=1.5):
    """Boîte arrondie avec label centré et sous-label optionnel en italique."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.025',
        fc=fc, ec=C['edge'], lw=lw, alpha=a, zorder=3,
    ))
    cx, cy = x + w / 2, y + h / 2
    dy = 0.05 if sub else 0
    ax.text(cx, cy + dy, label,
            ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.06, sub,
                ha='center', va='center', fontsize=max(fs - 2.5, 8),
                color=tc, alpha=0.88, style='italic', zorder=4)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 : Bloc-diagramme de la structure du circuit miroir
# ════════════════════════════════════════════════════════════════════════════
def fig_mirror_schematic():
    fig, ax = plt.subplots(figsize=(16, 4.5), facecolor=C['bg'])
    ax.set(xlim=(0, 16), ylim=(0, 1), facecolor=C['bg'])
    ax.axis('off')

    ax.text(8, 0.97,
            'Circuit miroir Pauli-randomisé  (Proctor et al., PRL 2022)',
            ha='center', va='top', fontsize=16, fontweight='bold', color=C['text'])

    # Fils de qubits (3 qubits illustratifs)
    for qy in [0.28, 0.50, 0.72]:
        ax.hlines(qy, 0.3, 14.2, colors=C['wire'], lw=1.5, zorder=1)
    for i, qy in enumerate([0.72, 0.50, 0.28]):
        ax.text(0.05, qy, f'q{i}', va='center', ha='left',
                fontsize=11, color=C['text'])

    # Blocs : (x, y, w, h, couleur, label, sous-label, couleur texte)
    bh, by = 0.78, 0.10
    BLOCKS = [
        (0.40,  by, 1.20, bh, C['fwd'],    'C₀',             'Clifford\ninitial',           'white'),
        (1.80,  by, 3.60, bh, C['pdress'], 'P · L  ×  d',    'd couches\nPauli + Clifford', C['text']),
        (5.60,  by, 1.80, bh, C['pmid'],   'P_mid',           'Paulis\naléatoires',          'white'),
        (7.60,  by, 3.60, bh, C['inv'],    "L† · P'  ×  d",  'd couches\nquasi-inverses',   'white'),
        (11.40, by, 1.20, bh, C['inv'],    'C₀†',            'Clifford\ninverse',            'white'),
        (12.85, by, 1.00, bh, C['meas'],   '⊠',              None,                          'white'),
    ]

    for bx, b_y, bw, b_h, bc, lbl, sub, tc in BLOCKS:
        rbox(ax, bx, b_y, bw, b_h, bc, lbl, sub=sub, tc=tc)

    # Axe miroir (ligne en tirets centrée sur P_mid)
    mid_x = 5.60 + 1.80 / 2  # 6.50
    ax.axvline(mid_x, 0.06, 0.94,
               color=C['pmid'], lw=2.5, ls='--', zorder=2)
    ax.text(mid_x, 0.965, 'axe miroir', ha='center', va='top',
            fontsize=9, color='#CC7A00', style='italic')

    # Surbrillance du bloc P_mid
    ax.add_patch(FancyBboxPatch(
        (5.55, 0.06), 1.90, 0.88,
        boxstyle='round,pad=0.04',
        fc='none', ec=C['pmid'], lw=3, alpha=0.6, zorder=2,
    ))

    # Annotation de sortie
    ax.text(13.35, 0.06, 'bitstring s ≠ |0…0⟩',
            ha='center', va='top', fontsize=10,
            color=C['inv'], fontweight='bold', style='italic')

    # Légende
    handles = [
        mpatches.Patch(fc=C['fwd'],    ec=C['edge'], label='Clifford initial C₀'),
        mpatches.Patch(fc=C['pdress'], ec=C['edge'], label='Couche P (Pauli) + L (Clifford)'),
        mpatches.Patch(fc=C['pmid'],   ec=C['edge'], label="P_mid : Paulis centraux (brise l'écho)"),
        mpatches.Patch(fc=C['inv'],    ec=C['edge'], label="Quasi-inverse L†, C₀†"),
        mpatches.Patch(fc=C['meas'],   ec=C['edge'], label='Mesure'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=9.5,
              framealpha=0.92, bbox_to_anchor=(0.0, -0.05))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'mirror_schematic.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print('  [1/5] mirror_schematic.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 : Comparaison annotée écho naïf vs circuit miroir
# ════════════════════════════════════════════════════════════════════════════
def fig_echo_vs_mirror():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=C['bg'])
    fig.suptitle('Écho naïf vs Circuit miroir Pauli-randomisé',
                 fontsize=17, fontweight='bold', color=C['text'], y=1.02)

    panels = [
        dict(
            title='Circuit écho naïf  U · U†',
            tcolor=C['bad'],
            blocks=[
                (0.07, 0.62, 0.38, 0.25, C['fwd'], 'U',    None, 'white'),
                (0.55, 0.62, 0.38, 0.25, C['bad'], 'U†',   None, 'white'),
            ],
            bullets=[
                ('U · U† = I  (toujours)', False),
                ('Sortie : |0…0⟩  peu importe le bruit', False),
                ('Taux de succès P ≡ 1', False),
                ('→  Insensible au bruit !', True),
            ],
            output='|0…0⟩  (trivial)',
        ),
        dict(
            title='Circuit miroir Pauli-randomisé',
            tcolor=C['inv'],
            blocks=[
                (0.04, 0.62, 0.26, 0.25, C['fwd'],  'U',      None, 'white'),
                (0.34, 0.62, 0.32, 0.25, C['pmid'], 'P_mid',  None, 'white'),
                (0.70, 0.62, 0.26, 0.25, C['inv'],  'U_q†',   None, 'white'),
            ],
            bullets=[
                ('U · P_mid · U_q† ≠ I  en général', False),
                ('Bitstring cible s = f(P_mid, U) unique', False),
                ('P(s) décroît proportionnellement au bruit', False),
                ('→  Sonde directe de la fidélité', True),
            ],
            output='s ≠ |0…0⟩',
        ),
    ]

    for ax, panel in zip(axes, panels):
        ax.set(xlim=(0, 1.08), ylim=(0, 1), facecolor=C['bg'])
        ax.axis('off')

        # Barre de titre
        rbox(ax, 0.02, 0.85, 0.96, 0.13, panel['tcolor'], panel['title'],
             tc='white', fs=13)

        # Blocs de circuit
        arrow_y = 0.745
        for bx, by, bw, bh, bc, lbl, sub, tc in panel['blocks']:
            rbox(ax, bx, by, bw, bh, bc, lbl, sub=sub, tc=tc, fs=13)

        # Flèches entre blocs
        for i in range(len(panel['blocks']) - 1):
            x0 = panel['blocks'][i][0] + panel['blocks'][i][2]
            x1 = panel['blocks'][i + 1][0]
            ax.annotate('', xy=(x1, arrow_y), xytext=(x0, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=1.8))

        # Flèche de sortie + label
        x_end = panel['blocks'][-1][0] + panel['blocks'][-1][2]
        ax.annotate('', xy=(0.99, arrow_y), xytext=(x_end, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.8))
        ax.text(1.00, arrow_y, panel['output'],
                va='center', ha='left', fontsize=10,
                color=panel['tcolor'], fontweight='bold', style='italic')

        # Séparateur
        ax.hlines(0.59, 0.02, 0.98, colors=C['sep'], lw=1.2)

        # Points de billet
        tc_bullet = '#C0392B' if panel['tcolor'] == C['bad'] else '#1D6A3A'
        for i, (text, bold) in enumerate(panel['bullets']):
            ax.text(0.05, 0.55 - i * 0.12, text, va='top',
                    fontsize=11.5 if not bold else 12,
                    color=panel['tcolor'] if bold else tc_bullet,
                    fontweight='bold' if bold else 'normal')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'echo_vs_mirror.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print('  [2/5] echo_vs_mirror.png')


# ════════════════════════════════════════════════════════════════════════════
# Figures 3 & 4 : Circuits Qiskit concrets (2 qubits)
# ════════════════════════════════════════════════════════════════════════════
def _verify_target(qc):
    """Retourne (bitstring le plus probable, est_déterministe) à bruit nul."""
    from qiskit.quantum_info import Statevector
    probs = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict()
    k = max(probs, key=probs.get)
    return k, probs[k] > 0.9999


# Style Qiskit-mpl : Paulis (twirls) en ambre, squelette Clifford en bleu.
_GATE_STYLE = {
    'displaycolor': {
        'x':    (C['pmid'], '#FFFFFF'),
        'y':    (C['pmid'], '#FFFFFF'),
        'z':    (C['pmid'], '#FFFFFF'),
        'sx':   (C['fwd'],  '#FFFFFF'),
        'sxdg': (C['fwd'],  '#FFFFFF'),
        'ry':   (C['fwd'],  '#FFFFFF'),
        'cx':   (C['fwd'],  '#FFFFFF'),
    }
}


def _build_skeleton(qc):
    """Squelette Clifford partagé : C₀ · L · L† · C₀†  (seed=13 mitiq)."""
    qc.sxdg(1); qc.ry(-np.pi / 2, 1)        # C₀
    qc.barrier()
    qc.cx(0, 1)                              # L
    qc.barrier()
    qc.cx(0, 1)                              # L†
    qc.barrier()
    qc.ry(np.pi / 2, 1); qc.sx(1)           # C₀†


def _build_mirror(qc):
    """Même squelette + couches de Paulis (twirls) : P_fwd, P_mid, P_inv."""
    qc.sxdg(1); qc.ry(-np.pi / 2, 1)        # C₀
    qc.barrier()
    qc.z(0); qc.x(1)                         # P_fwd  (twirl aller)
    qc.cx(0, 1)                              # L
    qc.barrier()
    qc.x(0); qc.z(1)                         # P_mid  (twirl central)
    qc.barrier()
    qc.cx(0, 1)                              # L†
    qc.x(0); qc.y(1)                         # P_inv  (twirl retour)
    qc.barrier()
    qc.ry(np.pi / 2, 1); qc.sx(1)           # C₀†


def _draw(qc, size):
    try:
        return qc.draw('mpl', style=_GATE_STYLE, fold=-1, initial_state=True)
    except Exception:
        return qc.draw('mpl', fold=-1, initial_state=True)


def fig_qiskit_circuits():
    """Deux circuits partageant EXACTEMENT le même squelette Clifford :
       - écho = squelette seul                → sortie |00⟩ (triviale),
       - miroir = squelette + Paulis (twirls) → sortie |10⟩ (non-triviale).
       Les deux sorties sont vérifiées par simulation statevector.
       Les Paulis ajoutés sont surlignés en ambre pour la lisibilité.
    """
    from qiskit import QuantumCircuit, ClassicalRegister

    # ── Écho : squelette Clifford seul (mêmes portes, sans Paulis) ───────────
    qc_echo = QuantumCircuit(2)
    _build_skeleton(qc_echo)
    echo_key, echo_det = _verify_target(qc_echo)
    qc_echo.add_register(ClassicalRegister(2, 'c'))
    qc_echo.measure([0, 1], [0, 1])
    assert echo_det and echo_key == '00', f'écho inattendu: {echo_key}, det={echo_det}'

    f_echo = _draw(qc_echo, None)
    f_echo.set_size_inches(13, 3)
    f_echo.suptitle(
        f'① Écho : squelette Clifford seul  (C₀ · L · L† · C₀†)   →   '
        f'sortie |{echo_key}⟩  (triviale)',
        fontsize=13, fontweight='bold', y=1.06, color=C['bad'])
    p1 = os.path.join(OUT_DIR, 'circuit_echo_qiskit.png')
    f_echo.savefig(p1, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(f_echo)
    print(f'  [3/5] circuit_echo_qiskit.png   (sortie vérifiée |{echo_key}⟩)')

    # ── Miroir : MÊME squelette + Paulis (twirls) surlignés ──────────────────
    qc_mir = QuantumCircuit(2)
    _build_mirror(qc_mir)
    mir_key, mir_det = _verify_target(qc_mir)
    qc_mir.add_register(ClassicalRegister(2, 'c'))
    qc_mir.measure([0, 1], [0, 1])
    assert mir_det and mir_key != '00', f'miroir non valide: {mir_key}, det={mir_det}'

    f_mir = _draw(qc_mir, None)
    f_mir.set_size_inches(13, 3)
    f_mir.suptitle(
        f'② Miroir : même squelette + Paulis (twirls, en ambre)   →   '
        f'sortie |{mir_key}⟩  (non-triviale, vérifiée)',
        fontsize=13, fontweight='bold', y=1.06, color=C['inv'])
    p2 = os.path.join(OUT_DIR, 'circuit_mirror_qiskit.png')
    f_mir.savefig(p2, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(f_mir)
    print(f'  [4/5] circuit_mirror_qiskit.png  (sortie vérifiée |{mir_key}⟩, '
          f'seule différence = Paulis ambre)')


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 : Décroissance de P(s) — écho ≡ 1 vs miroir qui décroît
# ════════════════════════════════════════════════════════════════════════════
def fig_success_prob_decay():
    depths = np.arange(0, 15)
    p_gate   = 0.02       # 2% erreur par porte par couche
    n_qubits = 2
    # Taux de succès miroir : modèle simplifié P(s) ≈ (1 - p_eff)^d
    p_eff  = 1 - (1 - p_gate) ** (n_qubits * 2)
    decay  = (1 - p_eff) ** depths
    floor  = 1 / (2 ** n_qubits)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C['bg'])
    fig.suptitle(
        'Taux de succès P(s) en fonction de la profondeur du circuit',
        fontsize=16, fontweight='bold', color=C['text'], y=1.02,
    )

    # ── Écho naïf : toujours 1 ───────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(C['bg'])
    ax0.plot(depths, np.ones_like(depths), 'o-', color=C['bad'],
             lw=2.5, ms=8, label='P(|0…0⟩)  écho naïf', zorder=3)
    ax0.axhline(1.0, color=C['bad'], lw=1.5, ls='--', alpha=0.35)
    ax0.set(xlabel='Profondeur d', ylabel='P(|0…0⟩)',
            ylim=(0, 1.2), xlim=(-0.5, 14))
    ax0.set_title('Écho naïf', fontsize=14, fontweight='bold', color=C['bad'])
    ax0.text(7, 0.52,
             'Bruit invisible\nquelle que soit l\'erreur de porte',
             ha='center', fontsize=13, color=C['bad'], fontweight='bold',
             bbox=dict(fc='#FDECEA', ec=C['bad'], boxstyle='round,pad=0.4', alpha=0.9))
    ax0.legend(fontsize=11)
    for spine in ['top', 'right']:
        ax0.spines[spine].set_visible(False)

    # ── Circuit miroir : décroissance exponentielle ──────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor(C['bg'])
    ax1.plot(depths, decay, 's-', color=C['inv'],
             lw=2.5, ms=8, label='P(s)  circuit miroir', zorder=3)
    ax1.axhline(floor, color='#AAAAAA', lw=1.5, ls='--',
                label=f'Plancher aléatoire 1/2ⁿ = {floor:.2f}')
    ax1.set(xlabel='Profondeur d', ylabel='P(s)',
            ylim=(0, 1.2), xlim=(-0.5, 14))
    ax1.set_title('Circuit miroir Pauli-randomisé',
                  fontsize=14, fontweight='bold', color=C['inv'])
    ax1.annotate('Pente → fidélité\ndes portes logiques',
                 xy=(6, float(decay[6])),
                 xytext=(9.5, float(decay[6]) + 0.27),
                 fontsize=12, color=C['inv'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C['inv'], lw=1.5))
    ax1.legend(fontsize=11)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'success_prob_decay.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print('  [5/5] success_prob_decay.png')


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Génération des figures dans : {OUT_DIR}\n')
    fig_mirror_schematic()
    fig_echo_vs_mirror()
    fig_qiskit_circuits()
    fig_success_prob_decay()
    print('\nTerminé — 5 figures générées.')
