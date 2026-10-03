"""One bounded continuation for the three reviewed AppleDouble decode failures."""
import json
import msvcrt
import os
import time
from dataset_upload_worker import MGT, STATE, run

ALLOWED = {
    'ExDark/archives/bicycle.tar.gz',
    'ExDark/archives/boat.tar.gz',
    'ExDark/archives/bottle.tar.gz',
}


def main():
    deadline = time.monotonic() + 4 * 3600
    print('REVIEWED_RETRY_WAITER', os.getpid(), flush=True)
    while time.monotonic() < deadline:
        with (MGT / 'upload_worker.lock').open('a+b') as lock:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                pass
            else:
                try:
                    state = json.loads(STATE.read_text(encoding='utf-8'))
                    failed = {k: v for k, v in state['results'].items()
                              if v.get('status') == 'failed'}
                    for key, result in failed.items():
                        error = result.get('error', '')
                        if (key not in ALLOWED or 'UnidentifiedImageError' not in error
                                or '/._' not in error):
                            raise RuntimeError('Unreviewed failure: ' + key)
                    if state.get('status') != 'batch_finished':
                        raise RuntimeError('Previous batch did not finish normally: '
                                           + str(state.get('status')))
                    print('RETRY_REVIEWED_AND_REMAINING', sorted(failed), flush=True)
                    # Hold the same worker lock throughout: no concurrent state writers.
                    # run() rechecks local/remote hashes and skips completed files.
                    run()
                    final = json.loads(STATE.read_text(encoding='utf-8'))
                    remaining = [k for k, v in final['results'].items()
                                 if v.get('status') == 'failed']
                    if remaining:
                        raise RuntimeError('Retry requires review: ' + repr(remaining))
                    print('CONTINUATION_FINISHED_FINAL_AUDIT_STILL_REQUIRED', flush=True)
                    return
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        time.sleep(30)
    raise TimeoutError('Bounded retry wait expired; inspect current worker')


if __name__ == '__main__':
    with (MGT / 'reviewed_exdark_retry.lock').open('a+b') as singleton:
        singleton.seek(0)
        msvcrt.locking(singleton.fileno(), msvcrt.LK_NBLCK, 1)
        main()
