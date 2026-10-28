"""Finite-batch task processing with a bounded queue and error propagation.

Python 3.13+, standard library only. One producer and one worker.
Results are retained in memory; this example expects a finite input iterable.
Shutdown wakes queue waiters; a running process(item) must still finish.
"""
import sys

if sys.version_info < (3, 13):
    raise RuntimeError('This example requires Python 3.13+ for Queue.shutdown')

from queue import Queue, ShutDown
from threading import Thread


def run_tasks(items, process, *, capacity=8):
    """Return ordered results on success; re-raise producer/worker errors.

    A producer exception takes precedence if both sides fail. Results from a
    failed run are not returned; already executed side effects are not undone.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError('capacity must be a positive integer')
    if not callable(process):
        raise ValueError('process must be callable')
    tasks = Queue(maxsize=capacity)
    results, errors = [], []

    def consume():
        while True:
            try:
                item = tasks.get()
            except ShutDown:
                return
            try:
                result = process(item)
            except BaseException as error:
                errors.append(error)
                tasks.shutdown(immediate=True)
                return
            else:
                results.append(result)
            finally:
                tasks.task_done()

    worker = Thread(target=consume, name='bounded-task-worker')
    worker.start()
    try:
        for item in items:
            try:
                tasks.put(item)
            except ShutDown:
                break
    except BaseException:
        tasks.shutdown(immediate=True)
        raise
    finally:
        tasks.shutdown()
        worker.join()
        tasks.join()
    if errors:
        raise errors[0]
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fail-at', type=int)
    args = parser.parse_args()

    def square(value):
        if value == args.fail_at:
            raise RuntimeError(f'processing task {value} failed')
        return value * value

    print(run_tasks(range(10), square, capacity=2))
