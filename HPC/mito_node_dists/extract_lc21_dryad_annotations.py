"""Extract author annotations for cohort bodies from the remote Dryad ZIP."""

import csv
import io
import pathlib
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args): return None
class RemoteZip(io.RawIOBase):
    def __init__(self):
        token = (pathlib.Path.home()/'.config/dryad/token').read_text().strip()
        request = urllib.request.Request('https://datadryad.org/api/v2/files/4505085/download', headers={'Authorization':'Bearer '+token})
        try:
            urllib.request.build_opener(NoRedirect).open(request,timeout=20).close()
        except urllib.error.HTTPError as error:
            if error.code != 302: raise RuntimeError(f'Dryad returned {error.code}') from None
            self.url=error.headers['Location']
        else:
            raise RuntimeError('Expected a redirect from Dryad')
        if urllib.parse.urlsplit(self.url).scheme != 'https':
            raise RuntimeError('Expected an HTTPS archive URL')
        self.cache=threading.local()
        self.offset=0
        self.tail,self.size=self.fetch('-65536')
    def fetch(self, interval):
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url,headers={'Range':'bytes='+interval}),timeout=35) as response:
                if response.status != 206: raise RuntimeError(f'Range returned {response.status}')
                return response.read(),int(response.headers['Content-Range'].split('/')[-1])
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Archive host returned {error.code}') from None
    def seekable(self): return True
    def tell(self): return self.offset
    def seek(self, value, whence=0):
        self.offset=(0,self.offset,self.size)[whence]+value
        return self.offset
    def read(self, count=-1):
        end=self.size if count<0 else min(self.size,self.offset+count)
        start=self.offset
        if end<=start: return b''
        cached=getattr(self.cache,'entry',None)
        if cached and cached[0]<=start<=end<=cached[0]+len(cached[1]):
            data=cached[1][start-cached[0]:end-cached[0]]
        elif start>=self.size-len(self.tail):
            data=self.tail[start-(self.size-len(self.tail)):end-(self.size-len(self.tail))]
        else:
            data,_=self.fetch(f'{start}-{end-1}')
        if len(data)!=end-start: raise RuntimeError('Incomplete ZIP range')
        self.offset=end
        return data


folder=pathlib.Path(sys.argv[1])
with (folder/'paper22_geometry.csv').open() as stream:
    bodies={row['body_id'] for row in csv.DictReader(stream)}
output=folder/'author_dryad'
output.mkdir(exist_ok=True)
patterns={
    'skeleton':re.compile(r'(?:^|/)saved_clean_skeletons/s_pandas_(\d+)_([^/]+)_200nm\.csv$'),
    'mito':re.compile(r'(?:^|/)saved_mito_df/([^/]+)_(\d+)_mito_df\.csv$'),
}
reader=RemoteZip()
with zipfile.ZipFile(reader) as archive:
    entries=sorted(archive.infolist(),key=lambda item:item.header_offset)
    end_offsets={a.header_offset:b.header_offset for a,b in zip(entries,entries[1:])}
    end_offsets[entries[-1].header_offset]=archive.start_dir
    selected=[]
    for info in entries:
        for category, pattern in patterns.items():
            match=pattern.search(info.filename)
            if match and match[1 if category=='skeleton' else 2] in bodies:
                selected.append(info)
        if info.filename.endswith('/neuron_quality.csv'):
            selected.append(info)
    names = {pathlib.PurePosixPath(info.filename).parts[-2:] for info in selected}
    if not selected or len(names) != len(selected):
        raise ValueError('No matching ZIP members or duplicate output paths')
    print(len(selected),'members;',sum(info.compress_size for info in selected),'compressed bytes',flush=True)

    def extract(info):
        parent,name=pathlib.PurePosixPath(info.filename).parts[-2:]
        path=output/parent/name
        if (path.is_file() and path.stat().st_size == info.file_size
                and zlib.crc32(path.read_bytes()) == info.CRC):
            return
        path.parent.mkdir(exist_ok=True)
        start=info.header_offset
        end=end_offsets[start]
        if end-start > info.compress_size+65536:
            raise RuntimeError(f'Unexpected record size for {name}')
        data,_=reader.fetch(f'{start}-{end-1}')
        reader.cache.entry=(start,data)
        try:
            payload=archive.read(info)
        finally:
            reader.cache.entry=None
        partial=path.with_suffix('.partial')
        partial.write_bytes(payload)
        partial.replace(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        failures=[]
        jobs=[(info,pool.submit(extract,info)) for info in selected]
        for index,(info,job) in enumerate(jobs,1):
            try:
                job.result()
            except (OSError,RuntimeError,zipfile.BadZipFile) as error:
                failures.append(info)
                print(f'Failed {pathlib.PurePosixPath(info.filename).name}: {type(error).__name__}',flush=True)
            if index%20==0 or index==len(selected):
                print(index,'/',len(selected),'checked;',len(failures),'failures',flush=True)
        if failures: raise SystemExit(1)
