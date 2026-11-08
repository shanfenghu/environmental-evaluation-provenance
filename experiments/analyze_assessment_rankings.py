"""Compare learner ordering under matched held-out and temporal score roles.

This inspected-record reanalysis is conditional on existing data and fitted
states. A configured inclusion comparison is expected to agree with the audit;
its agreement is continuity, not an independent scientific validation or speedup.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

TOLERANCE=1e-12


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expand(raw):
    """Keep raw losses and paired differences on their shared observations."""
    x=np.asarray(raw)
    return np.concatenate([x,(x[...,0]-x[...,2])[...,None],
                           (x[...,1]-x[...,2])[...,None],
                           (x[...,1]-x[...,0])[...,None]],axis=-1)


def score_roles(g):
    """Same scored blocks/weights; only the exclusion role changes."""
    g=np.asarray(g,dtype=float);k=len(g)
    if k<2 or g.shape!=(k,k,6) or not np.isfinite(g).all():
        raise ValueError('A finite square matrix of joint loss vectors is required')
    h=g[np.arange(k),np.arange(k)].mean(axis=0)
    t=(g.sum(axis=(0,1))-k*h)/(k*(k-1))
    return h,t


def direction(x):
    """Roundoff ties are explicit; tolerance is not practical equivalence."""
    x=np.asarray(x)
    return np.where(abs(x)<=TOLERANCE,0,np.sign(x)).astype(int)


def configured_inclusion(loss,ids,blocks,cells):
    """Equal-cell source inclusion scores; reconstruct each raw block score.

    Cells may differ in size. Remainders omitted from these bank scoring blocks
    stay fitted; no absent source's loss is imputed into the bank target.
    """
    loss=np.asarray(loss);k=len(blocks);pos={i:p for p,i in enumerate(ids)}
    owner={i:j for j,g in enumerate(blocks)for i in g}
    if len(owner)!=sum(map(len,blocks)) or set(owner)!=set(ids):
        raise ValueError('Every scored source must belong to exactly one block')
    if loss.shape!=(k,len(ids),3) or not np.isfinite(loss).all():
        raise ValueError('Source losses must match states, identities and learners')
    g=np.array([[np.mean([loss[j,[pos[i]for i in block if i in group]].mean(axis=0)
                            for group in cells.values()],axis=0)
                 for j in range(k)]for block in blocks])
    own=np.array([loss[owner[i],p]for p,i in enumerate(ids)])
    d=own-(loss.sum(axis=0)-own)/(k-1)
    baseline=np.mean([d[[pos[i]for i in ids if i in group]].mean(axis=0)
                      for group in cells.values()],axis=0)
    return expand(g),expand(baseline)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',default='records/pairing')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    REFERENCE=Path(args.records); OUTPUT=Path(args.output)
    started=time.perf_counter();run=json.loads((REFERENCE/'run.json').read_text())
    assert run['complete'];OUTPUT.mkdir(parents=True,exist_ok=False)
    record={'complete':False,
        'reference_run_sha256':sha(REFERENCE/'run.json'),
        'executor_sha256':sha(Path(__file__)),
        'scope':'Exploratory reanalysis of inspected source losses; no new assignments/fits/timings/environmental observations',
        'columns':['ridge','trees','seasonal','ridge_minus_seasonal',
                   'trees_minus_seasonal','trees_minus_ridge'],
        'primary_coordinate':5,'bootstrap_replicates':3999,
        'bootstrap_seed':2026110593,'tie_tolerance':TOLERANCE,'cases':[]}
    errors=[]
    for ci,ref in enumerate(run['cases']):
        p=REFERENCE/ref['record'];assert sha(p)==ref['sha256']
        case=json.loads(p.read_text());banks=[]
        for bank in case['banks']:
            p=REFERENCE/bank['loss_record'];assert sha(p)==bank['loss_record_sha256']
            with np.load(p,allow_pickle=False)as z:
                loss=np.array([z[s['loss_key']]for s in bank['states']])
            g,baseline=configured_inclusion(loss,bank['scored'],bank['blocks'],case['cells'])
            err=float(np.max(abs(g-np.array(bank['matrix']))));assert err<1e-10
            h,t=score_roles(g);d=h-t
            error=float(np.max(abs(d-baseline)));assert error<1e-10
            assert np.max(abs(d-np.array(bank['averaged'])))<1e-10
            errors.extend([err,error])
            dh,dt=direction(h),direction(t)
            banks.append({'bank':bank['bank'],'loss_sha256':sha(p),'heldout':h.tolist(),
                'temporal':t.tolist(),'audit':d.tolist(),'heldout_direction':dh.tolist(),
                'temporal_direction':dt.tolist(),'strict_reversal':(dh*dt==-1).tolist(),
                'configured_inclusion':baseline.tolist(),'matrix_error':err,'baseline_error':error})
        h=np.array([v['heldout']for v in banks]);t=np.array([v['temporal']for v in banks]);d=h-t
        draws=np.random.default_rng(2026110593+ci).integers(0,len(banks),(3999,len(banks)))
        summaries={}
        for name,x in [('heldout',h),('temporal',t),('audit',d)]:
            deleted=np.array([np.delete(x,i,axis=0).mean(axis=0)for i in range(len(banks))])
            summaries[name]={'mean':x.mean(axis=0).tolist(),'bank_min':x.min(axis=0).tolist(),
                'bank_max':x.max(axis=0).tolist(),
                'descriptive_95pct':np.quantile(x[draws].mean(axis=1),[.025,.975],axis=0).T.tolist(),
                'leave_one_bank_out_min':deleted.min(axis=0).tolist(),
                'leave_one_bank_out_max':deleted.max(axis=0).tolist()}
        summaries['strict_reversal_count']=(direction(h)*direction(t)==-1).sum(axis=0).tolist()
        summaries['heldout_ties']=(direction(h)==0).sum(axis=0).tolist()
        summaries['temporal_ties']=(direction(t)==0).sum(axis=0).tolist()
        summaries['heldout_positive_count']=(direction(h)==1).sum(axis=0).tolist()
        summaries['temporal_positive_count']=(direction(t)==1).sum(axis=0).tolist()
        item={'workload':case['workload'],'task':case['task'],'reference_case_sha256':sha(REFERENCE/ref['record']),
              'banks':banks,'summary':summaries}
        p=OUTPUT/f'case-{ci}.json';p.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
        record['cases'].append({'record':p.name,'sha256':sha(p),'workload':case['workload'],'task':case['task']})
        print(case['workload'],case['task'],'H/T/D',h.mean(axis=0)[3:],t.mean(axis=0)[3:],d.mean(axis=0)[3:],
              'reversals',summaries['strict_reversal_count'][3:],flush=True)
    record['maximum_reconstruction_error']=max(errors);record['analysis_seconds']=time.perf_counter()-started
    record['complete']=True;(OUTPUT/'run.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
