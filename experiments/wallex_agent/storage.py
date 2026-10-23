"""Atomic JSON snapshots safe for independent research and protective workers."""
from pathlib import Path
import json,os,tempfile


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as output:
            temporary=Path(output.name)
            json.dump(value,output,indent=2,sort_keys=True,allow_nan=False)
            output.write('\n');output.flush();os.fsync(output.fileno())
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
