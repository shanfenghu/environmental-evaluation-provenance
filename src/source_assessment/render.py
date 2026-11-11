"""Render existing assessment outcomes as legible stacked, single-column panels.

This is display reconstruction only. It does not fit models, choose settings,
resample records or change the recorded uncertainty calculations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np




def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(directory, output):
    """Render four current displays and neutral numeric CSVs from recorded results."""
    RESULTS=Path(directory)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    figures=output/'figures'; figures.mkdir()
    from .illustration import render_illustration
    illustration=render_illustration(figures/'operation.pdf')
    (output/'illustration.json').write_text(json.dumps(illustration,indent=2)+'\n')
    run = json.loads((RESULTS/'run.json').read_text())
    summary = json.loads((RESULTS/'summary.json').read_text())
    assert run['complete'] and len(summary['cases']) == 4
    cases = []
    for ref in run['cases']:
        path = RESULTS/ref['record']
        assert digest(path) == ref['sha256']
        cases.append(json.loads(path.read_text()))
    plt.rcParams.update({'font.size':8.5, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    colours = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', 's', '^']
    legend_labels = ['Ridge − seasonal', 'Trees − seasonal', 'Trees − ridge']
    receipt = {'run_sha256':digest(RESULTS/'run.json'),
               'summary_sha256':digest(RESULTS/'summary.json'),
               'case_inputs':run['cases'], 'font_points':8.5,
               'canvas_inches':[3.45,3.55], 'series_markers':markers, 'points':[]}
    for display in ['direct','order','confirmation']:
        fig,axes = plt.subplots(2,1,figsize=(3.45,3.55),layout='constrained')
        labels = legend_labels if display!='order' else ['Forward: fit first','Reverse: cohort first','Shapley average']
        for wi,ax in enumerate(axes):
            ax.axvline(0,color='.65',linewidth=.7)
            for ti in range(2):
                index=wi*2+ti
                case,item=cases[index],summary['cases'][index]
                for j,k in enumerate([3,4,5]):
                    y=(1-ti)+(.18-j*.18)
                    if display=='direct':
                        values=np.array([c['means']['swap_exclusion'][k] for c in case['cycles'][4:]])
                        v,lo,hi=values.mean(),values.min(),values.max()
                        ax.plot([lo,hi],[y,y],color=colours[j],linewidth=1)
                        ax.scatter(values,np.full(4,y),color=colours[j],marker=markers[j],s=7,alpha=.5)
                        ax.scatter([v],[y],color=colours[j],marker=markers[j],s=22,zorder=4)
                    elif display=='confirmation':
                        f=item['followup'];v=f['prediction_minus_confirmation'][k]
                        lo,hi=f['difference_interval_low'][k],f['difference_interval_high'][k]
                        ax.plot([lo,hi],[y,y],color=colours[j],linewidth=1.2)
                        ax.scatter([v],[y],color=colours[j],marker=markers[j],s=18,zorder=4)
                    else:
                        field=['forward','reverse','shapley'][j]
                        row=item['field_slices'][field][0]
                        v=item['joint']['mean'][row][5]
                        lo,hi=[item['joint'][name][row][5] for name in ['joint_band_low','joint_band_high']]
                        ax.plot([lo,hi],[y,y],color=colours[j],linewidth=1.2)
                        ax.scatter([v],[y],color=colours[j],marker=markers[j],s=18,zorder=4)
                    receipt['points'].append({'display':display,'case':index,'contrast':j,
                                              'mean':float(v),'low':float(lo),'high':float(hi),
                                              **({'cycle_values':values.tolist()} if display=='direct' else {})})
            unit='ppb' if wi==0 else 'mm/day'
            ax.set_yticks([1,0],['Dispersed','Edge'])
            ax.set_ylim(-.4,1.5)
            domain='US station NO$_2$' if wi==0 else 'GB specific discharge'
            ax.set_title(f'{domain} ({unit})' if display=='order' else domain,fontsize=9)
            title={'direct':'Matched exclusion contrast','order':'Training allocation: trees − ridge',
                   'confirmation':'Predicted − measured contrast'}[display]
            ax.set_xlabel(title if display=='order' else f'{title} ({unit})')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis='x',alpha=.15)
        handles=[plt.Line2D([],[],color=c,marker=marker,linestyle='none',markersize=4)
                 for c,marker in zip(colours,markers)]
        fig.legend(handles,labels,loc='outside upper center',ncol=1,frameon=False,fontsize=8)
        fig.savefig(figures/(display+'.pdf'),bbox_inches='tight',pad_inches=.02)
        plt.close(fig)
    (output/'display-data.json').write_text(json.dumps(receipt,indent=2)+'\n')
    from .tables import render_tables
    render_tables(results=RESULTS, output=output/'tables')

