"""Read-only public dataset metadata probes; does not download image archives."""
import concurrent.futures
import json
import re
import requests

PROXIES = {'https': 'http://127.0.0.1:7897', 'http': 'http://127.0.0.1:7897'}
TARGETS = {
    'NEU-DET': 'https://drive.google.com/file/d/1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k/view',
    'HRSID-JPG-inshore': 'https://drive.google.com/file/d/1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms/view',
    'RDD2020': 'https://data.mendeley.com/public-api/datasets/5ty2wb6gvg/versions/1/files',
    'D-Fire-linked-Kaggle': 'https://www.kaggle.com/api/v1/datasets/list?search=smoke-fire-detection-yolo',
    'GTSDB': 'https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html',
}

def probe(item):
    name, url = item
    try:
        response = requests.get(url, proxies=PROXIES, timeout=25)
        body = response.text
        if 'json' in response.headers.get('Content-Type', ''):
            data = response.json()
            if name == 'D-Fire-linked-Kaggle':
                data = [{k: x.get(k) for k in ('ref', 'title', 'totalBytes')} for x in data]
            return {'name': name, 'status': response.status_code, 'metadata': str(data)[:10000]}
        patterns = [r'.{0,100}fileSize.{0,150}', r'.{0,80}sizeBytes.{0,150}',
                    r'.{0,60}\b\d+(?:\.\d+)?\s*(?:GB|MB|GiB|MiB).{0,100}',
                    r'.{0,80}FullIJCNN.{0,100}', r'.{0,50}\.json.{0,100}']
        snippets = [s for p in patterns for s in re.findall(p, body)]
        return {'name': name, 'status': response.status_code, 'snippets': snippets[:25]}
    except Exception as exc:
        return {'name': name, 'error': str(exc)}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for result in pool.map(probe, TARGETS.items()):
            print(json.dumps(result, ensure_ascii=False))
    for name, ident in [('NEU-DET', '1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k'), ('HRSID-JPG-inshore', '1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms')]:
        try:
            with requests.get('https://drive.usercontent.google.com/download', params={'id': ident, 'export': 'download', 'confirm': 't'}, headers={'Range': 'bytes=0-0'}, stream=True, proxies=PROXIES, timeout=20) as r:
                print(json.dumps({'name': name, 'http_status': r.status_code, 'type': r.headers.get('Content-Type'), 'content_range': r.headers.get('Content-Range'), 'content_length': r.headers.get('Content-Length')}))
        except Exception as exc:
            print(json.dumps({'name': name, 'error': str(exc)}))
