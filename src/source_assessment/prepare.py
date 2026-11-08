"""Recreate pinned prepared inputs from provider archives under explicit terms.

Feature transforms are deterministic. Fitting and scoring periods are separate;
contemporary hydrological forcings support retrospective assessment, not an
operational forecast. Unknown upstream product ancestry is not reconstructed.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from .io import sha, save, contained


def scientific_members(item):
    """The pinned members actually used by preparation, excluding mutable wrappers."""
    return item.get('required_members', item['members'])


def verify_raw(directory, manifest, allow_container_repack=False):
    """Check exact containers or explicitly permitted identical scientific content.

    Repacking is never called recovery of the exact original ZIP. A changed used
    scientific member is rejected even when container repacking is permitted.
    """
    records=[]
    for item in manifest['inputs']:
        path=contained(directory,item['file'])
        exact=sha(path)==item['sha256']
        if not exact and not allow_container_repack:
            raise ValueError('Original archive bytes differ: '+item['file'])
        with zipfile.ZipFile(path) as archive:
            for member in scientific_members(item):
                content=archive.read(member['member'])
                if len(content)!=member['bytes'] or hashlib.sha256(content).hexdigest()!=member['sha256']:
                    raise ValueError('Scientific member content differs: '+member['member'])
        records.append({'file':item['file'],'exact_container':exact,'scientific_members_match':True})
    return records


def acquire(directory, manifest, allow_container_repack=False):
    """Download absent declared archives, retaining and verifying their actual bytes.

    Network availability is not guaranteed. Existing files are checked, never
    replaced. A temporary download is removed on failure, not installed silently.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for item in manifest['inputs']:
        dest=contained(directory,item['file'])
        if dest.exists():continue
        temporary=dest.with_suffix(dest.suffix+'.download')
        try:
            with urllib.request.urlopen(item['url'],timeout=120) as response, temporary.open('xb') as target:
                while chunk:=response.read(1024*1024):target.write(chunk)
            # Validate all used members before making the download visible.
            one={'inputs':[{**item,'file':temporary.name}]}
            verify_raw(directory,one,allow_container_repack)
            temporary.rename(dest)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return verify_raw(directory,manifest,allow_container_repack)


def features(date,lat,lon,hydro=None):
    """Fixed coordinate/seasonal terms; optional already declared forcing features."""
    day=pd.to_datetime(date).dayofyear.to_numpy()
    a=2*np.pi*day/365.25
    u=(lat-40)/10;v=(lon+20)/50;n=len(day)
    u=np.broadcast_to(u,n);v=np.broadcast_to(v,n)
    cols=[np.ones(n),u,v,u*u,v*v,u*v,np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),u*np.sin(a),u*np.cos(a),v*np.sin(a),v*np.cos(a)]
    if hydro is not None:cols.extend(hydro)
    return np.column_stack(cols).astype('float32')


def prepare_air(directory,config):
    """Select one training-completeness POC and fixed-coordinate daily site records."""
    frames=[]
    for name in config['archives']:
        with zipfile.ZipFile(contained(directory,name)) as archive:
            frames.append(pd.read_csv(archive.open(archive.namelist()[0]),
                dtype={'State Code':str,'County Code':str,'Site Num':str,'POC':str}))
    data=pd.concat(frames)
    data=data[data.Latitude.between(*config['latitude_bounds']) &
              data.Longitude.between(*config['longitude_bounds']) &
              (data['Observation Count']>=config['minimum_observations_per_day'])].copy()
    data['entity']=data['State Code']+'-'+data['County Code']+'-'+data['Site Num']
    data['date']=pd.to_datetime(data['Date Local']);arrays={};meta=[]
    for entity,g in data.groupby('entity',sort=True):
        earlier=g[g.date.dt.year==config['years'][0]];counts=earlier.groupby('POC').size()
        if counts.empty:continue
        poc=sorted(counts.index,key=lambda p:(-counts[p],p))[0]
        g=g[g.POC==poc].sort_values('date')
        if g.date.duplicated().any() or len(g[['Latitude','Longitude']].drop_duplicates())!=1:continue
        lat,lon=map(float,g.iloc[0][['Latitude','Longitude']]);dates=g.date.to_numpy().astype('datetime64[D]')
        xx=features(dates,lat,lon);yy=g['Arithmetic Mean'].to_numpy()
        valid=np.isfinite(yy)&np.isfinite(xx).all(axis=1)
        train=(g.date.dt.year.to_numpy()==config['years'][0])&valid
        later=(g.date.dt.year.to_numpy()==config['years'][1])&valid
        if min(train.sum(),later.sum())<config['minimum_days_each_period']:continue
        i=len(meta)
        for prefix,value in [('x',xx[train]),('y',yy[train]),('z',xx[later]),('w',yy[later]),('t',dates[train].astype('int32')),('s',dates[later].astype('int32'))]:arrays[f'{prefix}{i}']=value
        meta.append({'id':entity,'latitude':lat,'longitude':lon,'cell':int(lat>=config['geo_cut_lat'])*2+int(lon>=config['geo_cut_lon']),'train_rows':int(train.sum()),'later_rows':int(later.sum()),'poc':poc})
    return arrays,meta


def prepare_hydrology(directory,config):
    """Stream original catchments, preserving the original transformations/order."""
    arrays={};meta=[]
    with zipfile.ZipFile(contained(directory,'camels_gb_2020.zip')) as archive:
        topo=pd.read_csv(archive.open('data/CAMELS_GB_topographic_attributes.csv'),dtype={'gauge_id':str}).set_index('gauge_id')
        names=[n for n in archive.namelist() if '/timeseries/' in n and n.endswith('.csv')]
        names.sort(key=lambda n:n.split('_timeseries_')[1].split('_')[0])
        for name in names:
            entity=name.split('_timeseries_')[1].split('_')[0];row=topo.loc[entity]
            lat,lon,area,elev=row[['gauge_lat','gauge_lon','area','elev_mean']].astype(float)
            if not np.isfinite([lat,lon,area,elev]).all() or area<=0:continue
            f=pd.read_csv(io.BytesIO(archive.read(name)))
            if f.date.duplicated().any():raise ValueError('Duplicated catchment dates')
            f=f.sort_values('date');p=f.precipitation
            extra=[np.log1p(p.to_numpy())/3,np.log1p(p.rolling(7,min_periods=7).mean().to_numpy())/3,np.log1p(p.rolling(30,min_periods=30).mean().to_numpy())/3,f.temperature.to_numpy()/10,f.pet.to_numpy()/3,np.full(len(f),np.log1p(area)/5),np.full(len(f),elev/500)]
            xx=features(f.date.to_numpy(),lat,lon,extra);yy=f.discharge_spec.to_numpy()
            dates=pd.to_datetime(f.date).to_numpy().astype('datetime64[D]')
            valid=np.isfinite(yy)&np.isfinite(xx).all(axis=1)&(yy>=0)
            train=(dates>=np.datetime64(config['train_start']))&(dates<=np.datetime64(config['train_end']))&valid
            later=(dates>=np.datetime64(config['test_start']))&(dates<=np.datetime64(config['test_end']))&valid
            ntrain=int((np.datetime64(config['train_end'])-np.datetime64(config['train_start']))/np.timedelta64(1,'D'))+1
            nlater=int((np.datetime64(config['test_end'])-np.datetime64(config['test_start']))/np.timedelta64(1,'D'))+1
            if train.sum()<math.ceil(config['minimum_coverage_each_period']*ntrain) or later.sum()<math.ceil(config['minimum_coverage_each_period']*nlater):continue
            i=len(meta)
            for prefix,value in [('x',xx[train]),('y',yy[train].astype('float64')),('z',xx[later]),('w',yy[later].astype('float64')),('t',dates[train].astype('int32')),('s',dates[later].astype('int32'))]:arrays[f'{prefix}{i}']=value
            meta.append({'id':entity,'latitude':float(lat),'longitude':float(lon),'cell':int(lat>=config['geo_cut_lat'])*2+int(lon>=config['geo_cut_lon']),'train_rows':int(train.sum()),'later_rows':int(later.sum()),'negative_train_targets':int((yy[train]<0).sum())})
    return arrays,meta


def prepare(raw_dir,config_path,acquisition_path,output,allow_container_repack=False):
    """Recreate both declared domains at a new output; never overwrite inputs."""
    config=json.loads(Path(config_path).read_text());manifest=json.loads(Path(acquisition_path).read_text())
    checks=verify_raw(raw_dir,manifest,allow_container_repack)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name in ['air_quality_2021_2022','hydrology']:
        arrays,meta=(prepare_hydrology if name=='hydrology' else prepare_air)(raw_dir,config[name])
        path=output/(name+'.npz');np.savez_compressed(path,**arrays)
        save(path.with_suffix('.json'),{'domain':'hydrology' if name=='hydrology' else 'air_quality','entities':meta,'sha256':sha(path),'config':config[name],'source_boundary':'Retained daily products and fixed local transformations; unknown upstream measurement ancestry is not certified.'})
        print('Prepared',name,len(meta),'sources',flush=True)
    save(output/'preparation-verification.json',checks)
