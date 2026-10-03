"""Publish final reports only after the independent end-state audit passes."""
import base64
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import time

from dataset_upload_worker import MGT, REMOTE, SSH, sha, save

STATE = MGT / 'DELIVERY_CLOSER_STATE_20260921.json'
SOURCE_INDEX = MGT / 'SERVER_DATASET_INDEX_20260921.md'
FINAL_INDEX = MGT / 'SERVER_DATASET_INDEX_FINAL_20260921.md'
FINAL_REPORT = MGT / 'FINAL_COMPLETION_REPORT_20260921.md'


def checked_evidence():
    manifest = json.loads((MGT / 'UPLOAD_MANIFEST_20260921.json').read_text(encoding='utf-8'))
    transfer = json.loads((MGT / 'UPLOAD_STATE_20260921.json').read_text(encoding='utf-8'))
    local = json.loads((MGT / 'FINAL_LOCAL_ORIGINALS_AUDIT.json').read_text(encoding='utf-8'))
    remote = json.loads((MGT / 'FINAL_TRANSFER_AUDIT.json').read_text(encoding='utf-8'))
    if manifest.get('pending'):
        raise ValueError('Manifest still has pending items')
    expected = {j['remote']: (j['bytes'], j['sha256']) for j in manifest['files']}
    completed = {k: (v.get('bytes'), v.get('sha256')) for k, v in transfer['results'].items()
                 if v.get('status') == 'complete'}
    if completed != expected:
        raise ValueError('Transfer state does not exactly cover manifest')
    if local.get('status') != 'all_local_originals_preserved_hash_verified':
        raise ValueError('Local originals audit did not pass')
    local_map = {Path(x['path']).resolve(): (x['bytes'], x['sha256']) for x in local['files']}
    wanted_local = {Path(j['local']).resolve(): (j['bytes'], j['sha256']) for j in manifest['files']}
    if local_map != wanted_local:
        raise ValueError('Local originals audit coverage mismatch')
    if remote.get('status') != 'all_manifest_files_and_extracted_content_verified' or remote.get('errors'):
        raise ValueError('Remote final audit did not pass')
    remote_map = {x['path']: (x['bytes'], x['sha256']) for x in remote['files']}
    if remote_map != expected:
        raise ValueError('Remote final audit coverage mismatch')
    if set(remote.get('datasets', {})) != {j['dataset'] for j in manifest['files']}:
        raise ValueError('Remote final audit dataset coverage mismatch')
    return manifest, remote


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def publish(remote_name, local_path, expected_existing=None):
    data = local_path.read_bytes()
    encoded = base64.b64encode(data).decode()
    fingerprint = hashlib.sha256(data).hexdigest()
    existing = repr(expected_existing)
    code = f"""
import base64,hashlib,os,sys
sys.path.insert(0,{str(REMOTE + '/_management')!r})
from dataset_remote_ingest import ROOT,OWNER,safe
assert (ROOT/'_management/OWNER.txt').read_text().strip()==OWNER
p=safe({remote_name!r});data=base64.b64decode({encoded!r});wanted={fingerprint!r};allowed={existing}
if p.exists():
    assert p.is_file() and not p.is_symlink()
    current=hashlib.sha256(p.read_bytes()).hexdigest()
    assert current==wanted or (allowed is not None and current==allowed), 'Existing report differs; preserve and inspect'
if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.delivery_tmp')
    with t.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==wanted
print(wanted)
"""
    payload = base64.b64encode(code.encode()).decode()
    result = subprocess.run(SSH + [f"python3 -c \"import base64;exec(base64.b64decode('{payload}'))\""],
                            capture_output=True, text=True, encoding='utf-8', check=True, timeout=120)
    if result.stdout.strip() != fingerprint:
        raise ValueError('Published report hash confirmation mismatch')
    return fingerprint


def close_delivery():
    manifest, audit = checked_evidence()
    initial = SOURCE_INDEX.read_text(encoding='utf-8')
    in_progress = '**当前为上传过程中的目录说明，不是全部完成声明。**'
    finished_line = ('**清单内全部数据已完成上传、原包哈希核对、安全解压、逐成员校验及独立逐图解码。** '
                     '详细机器可读证据位于 `_management/FINAL_TRANSFER_AUDIT.json`。')
    if in_progress not in initial:
        raise ValueError('Source index no longer has the reviewed in-progress marker')
    total = sum(j['bytes'] for j in manifest['files'])
    stamp = time.strftime('%Y-%m-%d %H:%M:%S %z')
    summary = (f"\n\n## 最终交付状态\n\n"
               f"- 完成时间（本机时区）：{stamp}\n"
               f"- 已核验数据集：{len(audit['datasets'])} 套；清单文件：{len(manifest['files'])} 个；原始/说明文件合计：{total:,} bytes。\n"
               f"- 本地原始文件：逐文件大小及SHA256复核通过。\n"
               f"- 服务器原始文件：逐文件大小及SHA256复核通过；压缩包成员CRC或SHA256、解压成员集合及实际图片重新解码通过。\n"
               f"- JODD仍无实际下载包，不在上述完成计数中，也没有用不明镜像代替。\n"
               f"- 训练未启动，GPU未使用，共享环境和其他进程未修改。\n")
    final_index = initial.replace(in_progress, finished_line, 1) + summary
    write_atomic(FINAL_INDEX, final_index)
    report = ('# 数据集上传与验收最终报告\n\n' + summary.lstrip() + '\n'
              '独立审计状态：`all_manifest_files_and_extracted_content_verified`，错误列表为空。\n\n'
              '各数据集的标注边界、版本差异和读取注意事项见服务器根目录README及各数据集VALIDATION/REVIEW文件；原始标签未为通过验收而修改。\n')
    write_atomic(FINAL_REPORT, report)
    initial_sha = sha(SOURCE_INDEX)
    index_sha = publish('README.md', FINAL_INDEX, initial_sha)
    report_sha = publish('_management/FINAL_COMPLETION_REPORT_20260921.md', FINAL_REPORT)
    return dict(status='final_reports_published_hash_verified', datasets=len(audit['datasets']),
                files=len(manifest['files']), bytes=total, readme_sha256=index_sha,
                report_sha256=report_sha, finished=time.time())


def main():
    state = dict(status='waiting_for_independent_final_audit', pid=os.getpid(), started=time.time())
    save(STATE, state)
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        finalizer = json.loads((MGT / 'FINALIZER_STATE_20260921.json').read_text(encoding='utf-8'))
        if finalizer.get('status') == 'final_audit_passed_agent_review_and_readme_pending':
            state.update(status='reviewing_and_publishing_final_reports', updated=time.time());save(STATE,state)
            state.update(close_delivery());save(STATE,state)
            print('FINAL REPORTS PUBLISHED; agent completion review still required', flush=True)
            return
        if finalizer.get('status') == 'failed_requires_review':
            raise RuntimeError('Finalizer failed: ' + finalizer.get('error', 'unknown'))
        time.sleep(45)
    raise TimeoutError('Bounded delivery closer wait expired; inspect current transfers')


if __name__ == '__main__':
    with (MGT / 'delivery_closer.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        except Exception as e:
            state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
            state.update(status='failed_requires_review',error=str(e),finished=time.time());save(STATE,state)
            raise
