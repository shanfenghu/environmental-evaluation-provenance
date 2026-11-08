"""Load exact prepared arrays without importing preparation or planning code."""
import json
from pathlib import Path
import numpy as np
from .io import sha


class Data:
    """Resident per-source earlier/later records for a declared environmental task.

    The prepared first feature is a constant; native recipes omit that column
    and use their own intercept. Responses transform only on the fitting side.
    All dates are retained for chronology and calendar-month seasonal summaries.
    """
    def __init__(self, name, directory, cohorts):
        self.name=name
        self.path=Path(directory)/(name+'.npz')
        self.metadata=json.loads(self.path.with_suffix('.json').read_text())
        if sha(self.path)!=self.metadata['sha256']:
            raise ValueError('Prepared input byte identity mismatch')
        self.meta=self.metadata['entities'];self.hydro=name=='hydrology'
        self.input_hash=sha(self.path)
        self.dev=set(range(len(self.meta)))-set(cohorts[name]['external'])
        self.cells={c:[i for i in self.dev if self.meta[i]['cell']==c]
                    for c in sorted({row['cell'] for row in self.meta})}
        with np.load(self.path,allow_pickle=False) as a:
            self.x=[a[f'x{i}'][:,1:].astype('float64') for i in range(len(self.meta))]
            self.raw=[a[f'y{i}'] for i in range(len(self.meta))]
            self.y=[np.log1p(v) if self.hydro else v for v in self.raw]
            self.z=[a[f'z{i}'][:,1:].astype('float64') for i in range(len(self.meta))]
            self.w=[a[f'w{i}'] for i in range(len(self.meta))]
            self.month=[a[f't{i}'].astype('datetime64[D]').astype('datetime64[M]').astype(int)%12
                        for i in range(len(self.meta))]
            self.future=[a[f's{i}'].astype('datetime64[D]').astype('datetime64[M]').astype(int)%12
                         for i in range(len(self.meta))]
            self.last_training_day=max(int(a[f't{i}'].max()) for i in range(len(self.meta)))
            self.first_scoring_day=min(int(a[f's{i}'].min()) for i in range(len(self.meta)))
        if self.last_training_day>=self.first_scoring_day:
            raise ValueError('Fitting/scoring chronology is not separated')
