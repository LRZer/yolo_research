"""Download only the requested ZeroWaste-f official archive to the local E drive."""
import os
os.environ['DATASET_ROOT'] = 'E:/datasets'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
import shutil
import time
from dataset_acquire import ROOT, fetch, get, save

def main():
    dataset_dir = ROOT / 'ZeroWaste-f'
    dataset_dir.mkdir(exist_ok=True)
    state = dataset_dir / 'DOWNLOAD_RUN.json'
    save(state, {'status': 'initializing', 'pid': os.getpid(), 'started': time.time()})
    try:
        metadata = get('https://zenodo.org/api/records/6412647')
        save(dataset_dir / 'source_metadata.json', metadata)
        selected = next(f for f in metadata['files'] if f['key'] == 'zerowaste-f-final.zip')
        if selected['size'] != 7518242799 or selected['checksum'] != 'md5:e26e31a58080bca6782dca0e56074c5d':
            raise ValueError('Publisher metadata changed; manual review required')
        if shutil.disk_usage(ROOT).free < selected['size'] + 20_000_000_000:
            raise RuntimeError('Insufficient free space with 20GB safety reserve')
        job = dict(dataset='ZeroWaste-f', name=selected['key'], url=selected['links']['self'],
                   size=selected['size'], checksum=selected['checksum'], zip_password='UP#1VuX409z4')
        save(dataset_dir / 'download_manifest.json', [job])
        save(state, {'status': 'downloading', 'pid': os.getpid(), 'started': time.time(), 'job': job})
        result = fetch(job)
        save(state, {'status': result['status'], 'pid': os.getpid(), 'finished': time.time(), 'result': result})
        report = ('# ZeroWaste-f 本机下载状态\n\n'
                  f'- 状态：{result["status"]}\n'
                  '- 官方源：https://zenodo.org/records/6412647\n'
                  '- 文件：archives/zerowaste-f-final.zip\n'
                  '- 预期字节：7518242799\n'
                  '- 官方MD5：e26e31a58080bca6782dca0e56074c5d\n'
                  '- 官方解压密码：UP#1VuX409z4\n'
                  f'- 包结构检查：{result.get("archive_integrity", "尚未完成")}\n'
                  f'- 错误：{result.get("error", "无")}\n'
                  '- 本流程只下载和验证压缩包；未解压、未全量逐图解码/核对标注、未转换YOLO、未训练。\n')
        (dataset_dir / 'DOWNLOAD_STATUS.md').write_text(report, encoding='utf-8')
    except Exception as exc:
        save(state, {'status': 'failed', 'pid': os.getpid(), 'finished': time.time(), 'error': str(exc)})
        raise

if __name__ == '__main__':
    main()
