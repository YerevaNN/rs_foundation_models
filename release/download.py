"""Download pinned release assets and verify their recorded checksums (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def verify(path, size, checksum):
    if path.stat().st_size != size:
        raise ValueError(f'Wrong size: {path}')
    h = hashlib.new(checksum['type'].lower().replace('-', ''))
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest().lower() != checksum['value'].lower():
        raise ValueError(f'Checksum mismatch: {path}')

def download(item, destination):
    if item.get('restricted'):
        raise ValueError(f"Restricted asset: {item['name']}")
    checksum = item.get('checksum')
    if not checksum:
        raise ValueError(f"Missing checksum: {item['name']}")
    path = destination / item.get('directory','') / item['name']
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Unsafe asset path')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify(path, item['bytes'], checksum)
        return
    partial = path.with_name(path.name + '.partial')
    with urllib.request.urlopen(item['url'], timeout=120) as response, partial.open('wb') as out:
        while block := response.read(8 * 1024 * 1024):
            out.write(block)
    verify(partial, item['bytes'], checksum)
    partial.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', help='Exact name in datasets.json, or all')
    parser.add_argument('--chivit', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--list', action='store_true', help='List assets without downloading')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if bool(args.dataset) == args.chivit:
        parser.error('Choose --dataset or --chivit')
    if args.chivit:
        m = json.loads((base/'chivit.json').read_text())
        jobs = [(args.output, dict(name=m['filename'], bytes=m['bytes'], url=m['url'],
                                  checksum={'type':'SHA-256','value':m['sha256']}))]
    else:
        datasets = json.loads((base/'datasets.json').read_text())['datasets']
        selected = [d for d in datasets if args.dataset in ('all', d['name'])]
        if not selected:
            parser.error('Unknown dataset; see datasets.json')
        jobs = [(args.output/d['name'], f) for d in selected for f in d['files']]
    for destination, item in jobs:
        if args.list:
            print(f"{item['bytes']}\t{destination/item['name']}\t{item['url']}")
        else:
            download(item, destination)

if __name__ == '__main__':
    main()
