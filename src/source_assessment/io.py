"""Explicit, versioned input/output utilities."""
import hashlib
import json
from pathlib import Path


def sha(path):
    """SHA-256 of exact local bytes, without implying their scientific validity."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    """Write a finite JSON record to an explicitly chosen local destination."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contained(root, relative):
    """Reject absolute paths, parent traversal and escaping symbolic links."""
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Path escapes its declared root')
    path=root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes its declared root')
    return path
