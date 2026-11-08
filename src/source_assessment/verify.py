"""Independent weighting/role reconstruction and fixed native replay comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import sha, save


def expand(v):
    """Independent construction of the three paired learner differences."""
    x=np.asarray(v)
    return np.concatenate([x,np.stack([x[...,0]-x[...,2],x[...,1]-x[...,2],x[...,1]-x[...,0]],axis=-1)],axis=-1)


def verify(directory,data_dir,config_path,output):
    """Reconstruct all recorded scores without refitting; use complete runs only."""
    out = Path(directory)
    run = json.loads((out/'run.json').read_text())
    assert run['complete']
    assert sha(out/'losses.zip') == run['loss_archive_sha256']
    assert sha(config_path) == run['configuration_sha256']
    count, cycles, maximum, max_closure = 0, 0, 0., 0.
    cfg = json.loads(Path(config_path).read_text())
    assert len(run['cases']) == len(cfg['workloads'])*len(cfg['tasks'])
    support_checks = []
    seen, matched_pool_checks = set(), []
    summaries = {(c['workload'], c['task']): c for c in json.loads((out/'summary.json').read_text())['cases']}
    covariance_error = 0.
    with np.load(out/'losses.zip', allow_pickle=False) as archive:
        assert sorted(archive.files) == run['loss_archive_keys']
        assert len(archive.files) == len(run['cases'])*(1+cfg['cycles']*cfg['folds'])
        for ref in run['cases']:
            assert sha(out/ref['record']) == ref['sha256']
            case = json.loads((out/ref['record']).read_text())
            assert len(case['cycles']) == cfg['cycles']
            name = case['workload']
            assert (name, case['task']) not in seen
            seen.add((name, case['task']))
            meta_path = Path(data_dir)/(name+'.json')
            assert json.loads(meta_path.read_text())['sha256'] == case['input_sha256']
            assert sha(meta_path.with_suffix('.npz')) == case['input_sha256']
            meta = json.loads(meta_path.read_text())['entities']
            cells, quotas = case['cells'], case['quotas']
            dev, external = set(case['development']), set(case['external'])
            assert case['full_timing']['training_rows'] == sum(meta[i]['train_rows'] for i in dev)
            eligible = set().union(*map(set, cells.values()))
            assert dev == eligible | set(case['protected'])
            assert not dev & external
            assert case['latest_training_date'] < case['earliest_scoring_date']
            if case['task'] == 'northern_edge':
                expected = set()
                for c in cells:
                    ids = [i for i, row in enumerate(meta) if row['cell'] == int(c)]
                    expected.update(sorted(ids, key=lambda i: (-meta[i]['latitude'], meta[i]['id']))[:len(ids)//5])
                assert expected == external
            index = {i: j for j, i in enumerate(case['scored'])}
            support = []
            for c, ids in cells.items():
                source_lat = [meta[i]['latitude'] for i in ids]
                target_lat = [meta[i]['latitude'] for i in external if meta[i]['cell'] == int(c)]
                lo = max(min(source_lat), min(target_lat))
                hi = min(max(source_lat), max(target_lat))
                support.append({'cell': c,
                    'source_latitude_range': [min(source_lat), max(source_lat)],
                    'target_latitude_range': [min(target_lat), max(target_lat)],
                    'latitude_ranges_disjoint': lo > hi})
            support_checks.append({'workload': name, 'task': case['task'], 'cells': support,
                'interpretation': ('Shared latitude support is not observed; retain DISDE plug-in values only as extrapolative sensitivity'
                    if any(s['latitude_ranges_disjoint'] for s in support)
                    else 'Latitude ranges overlap; this does not establish joint covariate overlap or validate classifier calibration')})
            def signature(fitted, selected, role, grouping):
                value = {'input': case['input_sha256'], 'recipe': case['recipe_version'],
                         'fit': sorted(fitted), 'score': sorted(selected), 'role': role,
                         'cells': grouping, 'weights': 'equal_cell_entity_day'}
                return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            all_cells = {c: [i for i in case['scored'] if meta[i]['cell'] == int(c)] for c in cells}
            assert signature(dev, eligible, 'temporal_reference', cells) == case['full_query_signatures']['F']
            assert signature(dev, external, 'external_holdout', all_cells) == case['full_query_signatures']['Q']
            def cell_scores(raw, selected):
                result = []
                for c in cells:
                    ids = sorted(i for i in selected if meta[i]['cell'] == int(c))
                    assert ids
                    weights = np.zeros(len(raw))
                    for i in ids:
                        weights[index[i]] = 1/len(ids)
                    result.append(weights @ raw)
                return np.array(result)
            full = archive[case['full_loss_key']]
            assert full.shape == (len(index), 3) and np.isfinite(full).all()
            f = cell_scores(full, eligible).mean(axis=0)
            q = cell_scores(full, external).mean(axis=0)
            np.testing.assert_allclose(expand(f), case['full_population'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(expand(q), case['target'], atol=1e-10, rtol=0)
            pi = np.array([quotas[c]/len(cells[c]) for c in cells])
            disde_parts = []
            for weights in case['disde']['cells']:
                lp = np.array([full[index[i]] for i in weights['source']])
                lq = np.array([full[index[i]] for i in weights['target']])
                wp, wq = np.array(weights['source_weights']), np.array(weights['target_weights'])
                a = np.sum(lp*wp[:, None], axis=0)/wp.sum()
                b = np.sum(lq*wq[:, None], axis=0)/wq.sum()
                disde_parts.append([lp.mean(axis=0)-a, a-b, b-lq.mean(axis=0)])
            disde = expand(np.mean(disde_parts, axis=0))
            np.testing.assert_allclose(disde, case['disde']['components'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(disde.sum(axis=0), expand(f-q), atol=1e-10, rtol=0)
            row_differences, cycle_vectors = [], []
            for cycle_index, cycle in enumerate(case['cycles']):
                assert len(cycle['folds']) == cfg['folds']
                assert cycle['cycle'] == cycle_index
                assert cycle['seed'] == cfg['seed']+100000*cfg['workloads'].index(name)+10000*cfg['tasks'].index(case['task'])+cycle_index
                cycles += 1
                rng = np.random.default_rng(cycle['seed'])
                permutations = {c: rng.permutation(ids).tolist() for c, ids in cells.items()}
                expected_masks = [sorted(i for c in cells for i in permutations[c][j*quotas[c]:(j+1)*quotas[c]]) for j in range(5)]
                assert [row['validation'] for row in cycle['folds']] == expected_masks
                recomputed = []
                for j, row in enumerate(cycle['folds']):
                    v = set(row['validation'])
                    assert v <= eligible and not v & set(case['protected'])
                    used = dev-v
                    assert not used & external
                    other_used = dev-set(cycle['folds'][(j+1)%5]['validation'])
                    assert len(used) == len(other_used)
                    assert row['timing']['training_rows'] == sum(meta[i]['train_rows'] for i in used)
                    assert row['swap_training_rows'] == sum(meta[i]['train_rows'] for i in other_used)
                    row_differences.append(row['timing']['training_rows']-row['swap_training_rows'])
                    for label, fitted, selected, role, grouping in [
                        ('H', used, v, 'entity_holdout', cells),
                        ('A', used, eligible, 'temporal_reference', cells),
                        ('L', dev, v, 'temporal_reference', cells),
                        ('M', used, external, 'external_holdout', all_cells),
                        ('matched_swap', dev-set(cycle['folds'][(j+1)%5]['validation']), v, 'temporal_reference', cells)]:
                        assert signature(fitted, selected, role, grouping) == row['query_signatures'][label], label
                    raw = archive[row['loss_key']]
                    other = archive[cycle['folds'][(j+1)%5]['loss_key']]
                    assert raw.shape == full.shape and np.isfinite(raw).all()
                    assert other.shape == full.shape and np.isfinite(other).all()
                    own_cell = cell_scores(raw, v)
                    population_cell = cell_scores(raw, eligible)
                    h, a = own_cell.mean(axis=0), population_cell.mean(axis=0)
                    l, m = cell_scores(full, v).mean(axis=0), cell_scores(raw, external).mean(axis=0)
                    delta = own_cell-cell_scores(other, v)
                    values = {'components': np.array([h-a, a-f, f-q]),
                              'forward': np.array([h-l, l-q]), 'reverse': np.array([m-q, h-m]),
                              'shapley': np.array([((h-l)+(m-q))/2, ((l-q)+(h-m))/2]),
                              'interaction': (h-l)-(m-q), 'gap': h-q,
                              'predicted_exclusion': np.mean((own_cell-population_cell)/(1-pi[:, None]), axis=0),
                              'swap_exclusion': delta.mean(axis=0),
                              'swap_coupling': np.mean((1-pi[:, None])*delta, axis=0)}
                    for key, value in values.items():
                        error = float(np.max(abs(expand(value)-np.array(row['values'][key]))))
                        maximum = max(maximum, error)
                        assert error < 1e-10, key
                    for field in ['components', 'forward', 'reverse', 'shapley']:
                        error = float(np.max(abs(values[field].sum(axis=0)-values['gap'])))
                        max_closure = max(max_closure, error)
                        assert error < 1e-10
                    recomputed.append(values)
                    count += 1
                for field in recomputed[0]:
                    np.testing.assert_allclose(expand(np.mean([v[field] for v in recomputed], axis=0)),
                                               cycle['means'][field], atol=1e-10, rtol=0)
                order = ['components', 'forward', 'reverse', 'shapley', 'interaction', 'gap',
                         'predicted_exclusion', 'swap_exclusion', 'swap_coupling']
                cycle_vectors.append(np.concatenate([expand(np.mean([v[field] for v in recomputed], axis=0)).reshape(-1, 6)
                                                     for field in order], axis=0))
            vectors = np.array(cycle_vectors)
            item = summaries[name, case['task']]
            np.testing.assert_allclose(vectors.mean(axis=0), item['joint']['mean'], atol=1e-10, rtol=0)
            flat = vectors.reshape(len(vectors), -1)
            residuals = flat-flat.mean(axis=0)
            covariance = residuals.T@residuals/(len(flat)*(len(flat)-1))
            covariance_error = max(covariance_error, float(np.max(abs(covariance-np.array(item['joint']['covariance_of_mean'])))))
            assert covariance_error < 1e-10
            matched_pool_checks.append({'workload': name, 'task': case['task'],
                'predictive_training_entities_per_fold': len(dev)-sum(quotas.values()),
                'scored_validation_entities_per_fold': sum(quotas.values()),
                'maximum_absolute_training_daily_row_count_difference': max(map(abs, row_differences))})
    assert seen == {(w, t) for w in cfg['workloads'] for t in cfg['tasks']}
    assert count == len(run['cases'])*cfg['cycles']*cfg['folds']
    result = ({'complete': True, 'cases': len(run['cases']), 'whole_cycles': cycles,
                      'fold_source_states': count, 'full_source_states': len(run['cases']),
                      'historical_source_fingerprints_recorded': len(run['source_sha256']),
                      'maximum_query_reconstruction_error': maximum, 'maximum_closure_error': max_closure,
                      'maximum_joint_covariance_reconstruction_error': covariance_error,
                      'latitude_support_checks': support_checks,
                      'matched_pool_checks': matched_pool_checks,
                      'loss_archive_sha256': run['loss_archive_sha256'],
                      'scope': 'Reconstruction of retained native losses, roles, masks, weighting, allocations and cycle means; no all-fit replay or population guarantee'})
    save(output, result)
    return result


def compare(reference,replay,output,atol=1e-10):
    """Compare every replayed state's losses and non-timing scientific records.

    A reduced replay compares only its explicitly recorded original cycles.
    It does not imply that omitted cycles were refitted or bootstrap coverage
    was validated. Metadata-byte fingerprints differ after neutralisation.
    """
    reference,replay=Path(reference),Path(replay)
    expected=json.loads((reference/'run.json').read_text())
    actual=json.loads((replay/'run.json').read_text())
    assert actual['complete'] and expected['complete']
    assert actual['configuration_sha256']==expected['configuration_sha256']
    maximum=0.;states=0;cycles=0
    def compare_value(a,b,path):
        nonlocal maximum
        if isinstance(a,dict):
            assert set(a)==set(b),path
            for k in a:
                if k in ['metadata_sha256','elapsed_seconds_diagnostic_only','fit_seconds','score_seconds']:continue
                compare_value(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):compare_value(x,y,path+'/'+str(i))
        elif isinstance(a,(float,int)) and not isinstance(a,bool):
            error=abs(a-b);maximum=max(maximum,error)
            if error>atol:raise AssertionError(f'{path}: numerical discrepancy {error}')
        else:assert a==b,path
    with np.load(reference/'losses.zip',allow_pickle=False) as old, np.load(replay/'losses.zip',allow_pickle=False) as new:
        for key in new.files:
            np.testing.assert_allclose(new[key],old[key],rtol=0,atol=atol)
            maximum=max(maximum,float(np.max(abs(new[key]-old[key]))));states+=1
        for item in actual['cases']:
            a=json.loads((reference/item['record']).read_text());b=json.loads((replay/item['record']).read_text())
            ids={c['cycle'] for c in b['cycles']}
            a['cycles']=[c for c in a['cycles'] if c['cycle'] in ids]
            cycles+=len(ids)
            compare_value(a,b,item['record'])
    result={'complete':True,'stage':actual['stage'],'cases':len(actual['cases']),
            'whole_cycles_replayed':cycles,'native_source_states_replayed':states,
            'candidate_fits_replayed':2*states,'seasonal_fits_replayed':states,
            'reference_source_states':len(expected['loss_archive_keys']),
            'absolute_tolerance':atol,'maximum_absolute_numeric_difference':maximum,
            'excluded_from_comparison':['elapsed timing diagnostics','neutralised metadata byte fingerprints'],
            'full_native_run_replayed':set(actual['loss_archive_keys'])==set(expected['loss_archive_keys'])}
    save(output,result)
    return result
