"""Retrospective, fixed all-order analysis of the original source-loss bank."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from source_assessment.queries import FitScores, expand, query
from source_assessment.pairing import averaged_audit, paired_audit, interaction_matrix



def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',default='records/reference')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    SOURCE=Path(args.records); OUTPUT=Path(args.output)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    run=json.loads((SOURCE/'run.json').read_text())
    assert run['complete'] and sha(SOURCE/'losses.zip')==run['loss_archive_sha256']
    result={'scope':'Inspected-data reanalysis, not fresh data/assignment confirmation; exact enumeration of conditional fold order','source_loss_sha256':sha(SOURCE/'losses.zip'),'columns':run['columns'],'new_native_fits':0,'orders_per_five_block_bank':24,'source_sha256':{'analysis':sha(Path(__file__)),'pairing':sha(ROOT/'src/source_assessment/pairing.py')},'cases':[]}
    maximum=0.
    with np.load(SOURCE/'losses.zip',allow_pickle=False) as losses:
        for ref in run['cases']:
            case=json.loads((SOURCE/ref['record']).read_text())
            assert sha(SOURCE/ref['record'])==ref['sha256']
            cycles=[]
            for cycle in case['cycles']:
                folds=cycle['folds'];k=len(folds)
                matrix=np.empty((k,k,6))
                for j,fold in enumerate(folds):
                    state=FitScores(case['input_sha256'],case['recipe_version'],tuple(sorted(set(case['development'])-set(fold['validation']))),tuple(case['scored']),losses[fold['loss_key']])
                    for i,score in enumerate(folds):
                        val,_=query('block_pair',state,score['validation'],'entity_holdout' if i==j else 'temporal_reference',case['cells'],case['external'])
                        matrix[i,j]=expand(val)
                original=paired_audit(matrix)
                error=float(np.max(abs(original-np.array(cycle['means']['swap_exclusion']))))
                maximum=max(maximum,error);assert error<1e-12
                all_orders=np.array([paired_audit(matrix,(0,)+p) for p in itertools.permutations(range(1,k))])
                average=averaged_audit(matrix)
                assert np.max(abs(average-all_orders.mean(axis=0)))<1e-12
                centred=interaction_matrix(matrix)
                assert np.max(abs(average-k/(k-1)*np.mean(centred[np.arange(k),np.arange(k)],axis=0)))<1e-12
                cycles.append({'cycle':cycle['cycle'],'matrix':matrix.tolist(),'cyclic':original.tolist(),'averaged':average.tolist(),'all_orders':all_orders.tolist(),'minimum_over_orders':all_orders.min(axis=0).tolist(),'maximum_over_orders':all_orders.max(axis=0).tolist(),'conditional_order_covariance':np.cov(all_orders,rowvar=False,bias=True).tolist(),'conditional_order_sd':all_orders.std(axis=0).tolist()})
            old=np.array([x['cyclic'] for x in cycles]);new=np.array([x['averaged'] for x in cycles])
            cond=np.array([x['conditional_order_covariance'] for x in cycles])
            case_out={'workload':case['workload'],'task':case['task'],'cycles':cycles,'cyclic_mean':old.mean(axis=0).tolist(),'averaged_mean':new.mean(axis=0).tolist(),'cyclic_observed_cycle_sd':old.std(axis=0,ddof=1).tolist(),'averaged_observed_cycle_sd':new.std(axis=0,ddof=1).tolist(),'mean_removed_conditional_order_sd':np.sqrt(np.mean(np.diagonal(cond,axis1=1,axis2=2),axis=0)).tolist(),'confirmation_cyclic_mean':old[4:].mean(axis=0).tolist(),'confirmation_averaged_mean':new[4:].mean(axis=0).tolist(),'confirmation_minimum_over_all_orders':np.min([x['minimum_over_orders'] for x in cycles[4:]],axis=0).tolist(),'confirmation_maximum_over_all_orders':np.max([x['maximum_over_orders'] for x in cycles[4:]],axis=0).tolist()}
            result['cases'].append(case_out)
            print(case['workload'],case['task'],'confirmation averaged',np.round(new[4:].mean(axis=0),6).tolist(),'all-order min',np.round(case_out['confirmation_minimum_over_all_orders'],6).tolist(),'pairing SD',np.round(case_out['mean_removed_conditional_order_sd'],6).tolist())
    result['maximum_original_cyclic_reconstruction_error']=maximum
    (OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
