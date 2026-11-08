"""Exact payload inventory checks and safe standalone extraction."""
import json
from pathlib import Path
import stat
import zipfile
from .io import contained,sha


def verify_inventory(root,manifest_name='payload-manifest.json',require_data=True):
    """Reject changed, missing, additional or symlinked payload members.

    Data may be supplied by the separate prepared-input bundle. Use code-only
    mode only for inspecting that bundle split, not for native replay readiness.
    """
    root=Path(root)
    if root.is_symlink():raise ValueError('Symlinked payload root')
    manifest=json.loads(contained(root,manifest_name).read_text())
    declared={}
    for item in manifest['files']:
        path=contained(root,item['path'])
        name=path.relative_to(root.resolve()).as_posix()
        if name in declared:raise ValueError('Duplicate inventory path')
        declared[name]=item
        if item['bundle']=='inputs' and not require_data and not path.exists():continue
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Missing or modified payload member: '+name)
    found=set()
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Symlink in payload')
        if path.is_file():found.add(path.relative_to(root).as_posix())
    extra=found-set(declared)-{manifest_name}
    if extra:raise ValueError('Unclassified payload files: '+', '.join(sorted(extra)))
    return {'files':len(found),'expected_files':len(declared)+1,'require_data':require_data,'all_present_and_verified':require_data}


def safe_extract(archive,destination):
    """Extract only normal relative members into a new destination."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        seen=set()
        for info in z.infolist():
            name=info.filename
            if '\\' in name or name in seen or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe archive member')
            seen.add(name);contained(destination,name)
        z.extractall(destination)
    return destination
