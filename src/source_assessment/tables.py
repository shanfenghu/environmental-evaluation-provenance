"""Write neutral numeric tables without document-format dependencies."""
import csv
import json
from pathlib import Path


def write_csv(path, header, rows):
    """Write a single table with explicit columns and stable decimal strings."""
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def render_tables(results, output, pairing=None, profiles=None):
    """Render external scores, confirmation loss changes and follow-up summaries.

    Decimal rounding is display-only. The JSON records retain full precision;
    bootstrap ranges describe whole-bank resampling conditional on these data.
    """
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results/'summary.json').read_text())
    external, allocation, overlap, exclusion = [], [], [], []
    for item in summary['cases']:
        label = [item['workload'], item['task']]
        external.append(label+[f'{x:.3f}' for x in item['target'][:3]])
        measured = item.get('followup', {}).get('last_four_measured_exclusion')
        if measured is not None:
            # These are own-minus-other loss changes, not full-fit external MAEs.
            digits = 5 if item['workload'] == 'hydrology' else 4
            unit = 'mm/day' if item['workload'] == 'hydrology' else 'ppb'
            exclusion.append(label+['original_confirmation', 4, unit]+
                             [f'{x:+.{digits}f}' for x in measured[:3]])
        v = item['joint']['mean']
        allocation.append(label+[f'{v[j][5]:+.4f}' for j in range(3)]+
                          [f"{v[item['field_slices']['gap'][0]][5]:+.4f}"])
        if item['task'] == 'dispersed':
            overlap.append(label+[f'{row[5]:+.4f}' for row in item['disde']['components']])
        else:
            overlap.append(label+['No observed latitude overlap']*3)
    write_csv(output/'external.csv', ['workload','task','ridge','trees','seasonal'], external)
    if exclusion:
        write_csv(output/'exclusion-means.csv',
                  ['workload','task','panel','cycles','unit','ridge','trees','seasonal'], exclusion)
    write_csv(output/'allocation.csv', ['workload','task','coupling','refit','cohort','gap'], allocation)
    write_csv(output/'overlap.csv', ['workload','task','source_edge','conditional','target_edge'], overlap)
    if pairing is not None:
        analysis=json.loads(Path(pairing).read_text()); rows=[]
        for item in analysis['cases']:
            value=item['efficiency_ratio'][5]
            lo=item['intervals']['efficiency']['low'][5]
            hi=item['intervals']['efficiency']['high'][5]
            fmt=lambda x,d: f'{x:.{d}f}' if x is not None else 'Undefined'
            fraction=item['order_variance_fraction'][5]
            rows.append([item['workload'],item['task'],
                         fmt(None if fraction is None else 100*fraction,1),
                         fmt(item['fit_plus_query_time_ratio'],2),fmt(value,2),fmt(lo,2),fmt(hi,2),
                         hi is not None and hi<1])
        write_csv(output/'precision-cost.csv', ['workload','task','variance_removed_percent',
                  'fit_query_time_ratio','efficiency','descriptive_low','descriptive_high',
                  'unrounded_high_below_parity'], rows)
    if profiles is not None:
        directory=Path(profiles); run=json.loads((directory/'run.json').read_text()); rows=[]
        for ref in run['cases']:
            case=json.loads((directory/ref['record']).read_text()); s=case['summary']
            digits=4 if case['workload']=='hydrology' else 3
            lo,median,hi=s['quantiles10_50_90'][5]
            rows.append([case['workload'],case['task'],f"{100*s['positive_fraction'][5]:.1f}",
                         *[f'{x:.{digits}f}' for x in [median,lo,hi,s['top10pct_trimmed_mean'][5]]]])
        write_csv(output/'source-profiles.csv', ['workload','task','positive_percent','median',
                  'quantile10','quantile90','top10percent_trimmed_mean'], rows)
