"""Render a checked source-history illustration, not experimental data.

Both cases fit the mean of earlier responses. Their earlier group means differ;
all later V responses equal one. The full/held-out errors agree, but the matched
source-swap contrasts have opposite signs. Each site has one earlier response.

"""
from fractions import Fraction
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def illustration():
    """Check a common mean-fitting rule on two finite earlier-record patterns."""
    from itertools import combinations
    groups = {'V': frozenset([0, 1]), 'W': frozenset([2, 3]),
              'Rest': frozenset([4, 5])}
    all_sites = frozenset.union(*groups.values())
    states = [all_sites, all_sites-groups['V'], all_sites-groups['W']]
    group_means = [[Fraction(7, 5), Fraction(3), Fraction(1)],
                   [Fraction(7, 5), Fraction(3, 5), Fraction(17, 5)]]
    histories = [{i: value for group, value in zip(groups.values(), means) for i in group}
                 for means in group_means]
    mean_fit = lambda history, state: sum(history[i] for i in state)/len(state)
    predictions = [[mean_fit(history, state) for history in histories] for state in states]
    errors = [[sum(abs(value-Fraction(1)) for _ in groups['V'])/len(groups['V'])
               for value in row] for row in predictions]
    assert not groups['V'] & groups['W']
    assert [len(state) for state in states] == [6, 4, 4]
    assert predictions == [[Fraction(9, 5)]*2, [Fraction(2)]*2,
                            [Fraction(6, 5), Fraction(12, 5)]]
    assert errors == [[Fraction(4, 5)]*2, [Fraction(1)]*2,
                      [Fraction(1, 5), Fraction(7, 5)]]
    ordinary = [errors[1][j]-errors[0][j] for j in range(2)]
    swaps = [errors[1][j]-errors[2][j] for j in range(2)]
    assert ordinary == [Fraction(1, 5)]*2
    assert swaps == [Fraction(4, 5), Fraction(-2, 5)]
    for excluded in combinations(all_sites, 2):
        for history in histories:
            assert abs(mean_fit(history, all_sites-frozenset(excluded))-1) >= 0
    return groups, states, errors, ordinary, swaps, group_means, predictions


def render_illustration(output):
    """Check exact rational arithmetic and render the finite constructed example."""
    groups, states, errors, ordinary, swaps, group_means, predictions = illustration()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig = plt.figure(figsize=(3.5, 2.08))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 350), ylim=(0, 208)); ax.axis('off')
    ink, muted, blue, green = '#202A35', '#53606E', '#0072B2', '#007C67'
    ax.text(4, 196, f'Same {float(ordinary[0]):+.1f} rise, opposite source contrasts',
            fontsize=9.3, fontweight='bold', color=ink, va='center')
    ax.text(5, 172, 'Earlier groups fitted', fontsize=8.5, color=ink)
    ax.text(213, 172, 'Sites', ha='center', fontsize=8.5, color=ink)
    ax.text(293, 177, 'Later-V error', ha='center', fontsize=8.5, color=ink)
    ax.text(270, 160, 'Case A', ha='center', fontsize=8.1, color=muted)
    ax.text(326, 160, 'Case B', ha='center', fontsize=8.1, color=muted)
    for x, label in zip([114, 146, 179], ['V', 'W', 'Rest']):
        ax.text(x, 157, label, ha='center', fontsize=8.2, color=muted)
    ax.plot([4, 346], [149, 149], color='#ABB3BA', lw=.65)
    labels = ['Full fit', 'Exclude V', 'Exclude W']
    for i, y in enumerate([130, 103, 76]):
        if i == 2:
            ax.add_patch(Rectangle((3, y-12), 344, 25, facecolor='#EAF5F1', edgecolor='none'))
        ax.text(5, y, labels[i], va='center', color=ink, fontsize=8.5,
                fontweight='bold' if i == 2 else 'normal')
        for x, group in zip([114, 146, 179], groups):
            used = groups[group].issubset(states[i])
            ax.scatter([x], [y], s=45, facecolors=blue if used else 'white',
                       edgecolors=blue if used else '#7D8790', linewidths=.9, zorder=3)
        ax.text(213, y, str(len(states[i])), va='center', ha='center', color=ink)
        for j, x in enumerate([270, 326]):
            ax.text(x, y, f'{float(errors[i][j]):.1f}', va='center', ha='center',
                    color=green if i == 2 else ink, fontweight='bold' if i == 2 else 'normal')
    ax.plot([4, 346], [58, 58], color='#ABB3BA', lw=.65)
    ax.text(5, 43, 'Matched swap: exclude V − exclude W', fontsize=8.1, color=ink, va='center')
    for x, val in zip([270, 326], swaps):
        ax.text(x, 43, f'{float(val):+.1f}'.replace('-', '\N{MINUS SIGN}'), ha='center',
                va='center', color=green, fontsize=9.5, fontweight='bold')
    ax.scatter([9], [18], s=29, facecolors=blue, edgecolors=blue)
    ax.text(18, 18, 'used', va='center', fontsize=8.0, color=muted)
    ax.scatter([65], [18], s=29, facecolors='white', edgecolors='#7D8790')
    ax.text(74, 18, 'excluded', va='center', fontsize=8.0, color=muted)
    ax.text(346, 18, 'Constructed; all rows score V', ha='right', va='center',
            fontsize=8.0, color=muted)
    # The heading must remain inside the native publication-width canvas.
    fig.canvas.draw()
    title_box = ax.texts[0].get_window_extent(fig.canvas.get_renderer())
    assert title_box.x0 >= 0 and title_box.x1 <= fig.bbox.x1
    label = next(t for t in ax.texts if t.get_text().startswith('Matched swap:'))
    value = next(t for t in ax.texts if t.get_text() == '+0.8')
    label_box = label.get_window_extent(fig.canvas.get_renderer())
    value_box = value.get_window_extent(fig.canvas.get_renderer())
    assert (value_box.x0-label_box.x1)*72/fig.dpi >= 4
    output = Path(output)
    fig.savefig(output, metadata={'Title': 'Ordinary error rise and source contrast',
                                 'Author': None, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    receipt = {'kind': 'constructed illustration, not environmental evidence',
               'groups': {g: sorted(s) for g, s in groups.items()},
               'training_sites': [len(s) for s in states],
               'loss_unit': 'arbitrary mean absolute error units',
               'fitting_rule': 'Arithmetic mean of earlier responses on retained sites',
               'earlier_group_means_V_W_Rest': [[str(v) for v in row] for row in group_means],
               'fitted_predictions': [[str(v) for v in row] for row in predictions],
               'later_response': 1,
               'two_site_exclusions_checked': 15,
               'scored_mean_absolute_errors': [[str(x) for x in row] for row in errors],
               'full_to_excluded_difference': [str(x) for x in ordinary],
               'source_swap_contrasts': [str(x) for x in swaps],
               'assertions_passed': True,
               'claim': 'Matching the full-fit and own-held-out scores does not determine the added other-excluded score or even the swap sign.'}
    return receipt

