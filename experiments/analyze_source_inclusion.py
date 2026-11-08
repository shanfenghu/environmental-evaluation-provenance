"""Describe source-level fixed-bank exclusion contrasts, with explicit coverage.

Exploratory reanalysis of inspected losses. Quantiles and deletions are descriptive
and do not identify individual causal history effects or population uncertainty.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from source_assessment.queries import expand


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise(values, cells):
    """Equal-cell source summaries; missing coverage is not silently imputed."""
    groups = [sorted(set(ids)&set(values)) for ids in cells.values()]
    if any(not g for g in groups):
        raise ValueError('An observed cell is empty; cannot form the declared mean')
    x = np.array([values[i] for g in groups for i in g])
    w = np.array([1/(len(groups)*len(g)) for g in groups for i in g])
    assert abs(w.sum()-1) < 1e-12
    qs = []
    for coordinate in range(6):
        order = np.argsort(x[:,coordinate], kind='stable')
        cumulative = np.cumsum(w[order])
        qs.append([float(x[order[min(np.searchsorted(cumulative,p),len(order)-1)],coordinate])
                   for p in [.1,.5,.9]])
    trimmed, removed = [], []
    for g in groups:
        a = np.array([values[i] for i in g]); count = int(np.ceil(.1*len(g)))
        if count >= len(g):
            raise ValueError('No source remains after the prespecified deletion')
        trimmed.append(np.sort(a,axis=0)[:-count].mean(axis=0));removed.append(count)
    return {'observed_sources':len(values), 'observed_per_cell':[len(g)for g in groups],
            'source_mean':(x*w[:,None]).sum(axis=0).tolist(),
            'positive_fraction':((x>0)*w[:,None]).sum(axis=0).tolist(),
            'quantiles10_50_90':qs,
            'top10pct_trimmed_mean':np.mean(trimmed,axis=0).tolist(),
            'removed_per_cell':removed}


def source_average(records):
    gathered = {}
    for bank in records:
        for key,value in bank['values'].items():
            gathered.setdefault(int(key),[]).append(value)
    return ({i:np.mean(v,axis=0) for i,v in gathered.items()},
            {i:len(v) for i,v in gathered.items()})


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',default='records/pairing')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    REFERENCE=Path(args.records); OUTPUT=Path(args.output)
    run = json.loads((REFERENCE/'run.json').read_text());assert run['complete']
    OUTPUT.mkdir(parents=True,exist_ok=False)
    result = {'complete':False,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'src/source_assessment/queries.py',ROOT/'requirements.lock']},
        'reference_run_sha256':sha(REFERENCE/'run.json'),
        'columns':['ridge','boosting','seasonal','ridge_minus_seasonal',
                   'boosting_minus_seasonal','boosting_minus_ridge'],
        'primary_coordinate':'boosting_minus_ridge',
        'scope':'Exploratory source-level summaries on inspected fixed banks; no fresh assignment, fit, environmental confirmation or causal/coverage guarantee',
        'cases':[]}
    maximum = 0.
    for ci, ref in enumerate(run['cases']):
        case_path = REFERENCE/ref['record'];assert sha(case_path)==ref['sha256']
        case = json.loads(case_path.read_text());records=[]
        for bank in case['banks']:
            path = REFERENCE/bank['loss_record'];assert sha(path)==bank['loss_record_sha256']
            with np.load(path,allow_pickle=False) as archive:
                loss = np.array([archive[s['loss_key']] for s in bank['states']])
            ids = bank['scored'];owner={i:j for j,block in enumerate(bank['blocks']) for i in block}
            own = np.array([loss[owner[i],pos] for pos,i in enumerate(ids)])
            delta = expand(own-(loss.sum(axis=0)-own)/(len(bank['blocks'])-1))
            values = dict(zip(ids,delta))
            mean = np.mean([np.mean([values[i] for i in g if i in values],axis=0)
                            for g in case['cells'].values()],axis=0)
            discrepancy = float(np.max(abs(mean-np.array(bank['averaged']))))
            maximum=max(maximum,discrepancy);assert discrepancy < 1e-10
            records.append({'bank':bank['bank'],'loss_sha256':sha(path),
                'values':{str(i):v.tolist()for i,v in values.items()},
                'bank_mean':mean.tolist(),'reconstruction_discrepancy':discrepancy})
        values,counts = source_average(records);summary=summarise(values,case['cells'])
        eligible=set().union(*map(set,case['cells'].values()))
        summary.update({'eligible_sources':len(eligible), 'unobserved_sources':sorted(eligible-set(values)),
            'minimum_bank_coverage':min(counts.values()),'maximum_bank_coverage':max(counts.values()),
            'source_minus_bank_mean':(np.array(summary['source_mean'])-
                                     np.mean([x['bank_mean']for x in records],axis=0)).tolist()})
        deletions=[]
        for omitted in range(len(records)):
            av,cv=source_average([x for j,x in enumerate(records)if j!=omitted])
            s=summarise(av,case['cells'])
            deletions.append({'omitted_bank':omitted,'positive_fraction':s['positive_fraction'],
                              'trimmed_mean':s['top10pct_trimmed_mean'],
                              'minimum_coverage':min(cv.values()),'observed_sources':len(av)})
        target=OUTPUT/f'case-{ci}.json'
        item={'workload':case['workload'],'task':case['task'],'reference_case_sha256':sha(case_path),
              'cells':case['cells'],'summary':summary,'source_average':{
                  str(i):{'mean':v.tolist(),'banks':counts[i]}for i,v in values.items()},
              'banks':records,'leave_one_bank_out':deletions}
        target.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
        result['cases'].append({'record':target.name,'sha256':sha(target),
                               'workload':case['workload'],'task':case['task']})
        print(case['workload'],case['task'],json.dumps({k:summary[k]for k in [
            'observed_sources','minimum_bank_coverage','maximum_bank_coverage']}),
            'primary positive/quantiles/trimmed',summary['positive_fraction'][5],
            summary['quantiles10_50_90'][5],summary['top10pct_trimmed_mean'][5],flush=True)
    result['maximum_bank_reconstruction_discrepancy']=maximum
    result['complete']=True
    (OUTPUT/'run.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
