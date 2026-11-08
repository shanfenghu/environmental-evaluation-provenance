"""Payload, raw-content and path-boundary failure cases."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from source_assessment.artifact import safe_extract,verify_inventory
from source_assessment.prepare import verify_raw


class ArtifactTests(unittest.TestCase):
    def test_inventory_rejects_tampering_and_unclassified_files(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a').write_bytes(b'kept')
            manifest={'files':[{'path':'a','bytes':4,'sha256':hashlib.sha256(b'kept').hexdigest(),'bundle':'code'}]}
            (p/'payload-manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(verify_inventory(p)['all_present_and_verified'])
            (p/'other').write_text('unclassified')
            with self.assertRaises(ValueError):verify_inventory(p)
            (p/'other').unlink();(p/'a').write_bytes(b'edit')
            with self.assertRaises(ValueError):verify_inventory(p)

    def test_extract_rejects_parent_traversal_and_symbolic_links(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,member,mode in [(0,'../escape',0),(1,'link',0o120777<<16)]:
                arc=root/f'{n}.zip'
                with zipfile.ZipFile(arc,'w') as z:
                    info=zipfile.ZipInfo(member);info.external_attr=mode;z.writestr(info,b'../escape')
                with self.assertRaises(ValueError):safe_extract(arc,root/f'out-{n}')
            self.assertFalse((root/'escape').exists())

    def test_inventory_rejects_external_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'link').symlink_to('/etc/hosts')
            m={'files':[{'path':'link','bytes':1,'sha256':'x','bundle':'code'}]}
            (root/'payload-manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):verify_inventory(root)

    def test_repacked_container_is_explicit_and_changed_scientific_content_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'input.zip'
            def write(content):
                with zipfile.ZipFile(path,'w') as z:z.writestr('values.csv',content)
            write(b'1,2\n')
            m={'inputs':[{'file':'input.zip','sha256':'not-the-container',
                'members':[{'member':'values.csv','bytes':4,'sha256':hashlib.sha256(b'1,2\n').hexdigest()}]}]}
            with self.assertRaises(ValueError):verify_raw(root,m)
            self.assertFalse(verify_raw(root,m,True)[0]['exact_container'])
            write(b'9,9\n')
            with self.assertRaises(ValueError):verify_raw(root,m,True)


if __name__=='__main__':unittest.main()
