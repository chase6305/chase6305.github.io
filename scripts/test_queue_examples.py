#!/usr/bin/env python3
"""Run bounded queue failure paths in disposable processes with hard timeouts."""
import ast
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

if sys.version_info < (3, 13):
    raise SystemExit('Run queue shutdown checks with Python 3.13+')

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'content/posts/queue/bounded_tasks.py'
PREFIX = f'''import runpy, threading
from queue import Queue, ShutDown
space = runpy.run_path({str(EXAMPLE)!r})
run_tasks = space['run_tasks']
'''


class QueueExamples(unittest.TestCase):
    def run_child(self, source):
        result = subprocess.run([sys.executable, '-B', '-c', PREFIX + textwrap.dedent(source)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Exception in thread', result.stderr)

    def test_order_empty_and_one_shot_inputs(self):
        self.run_child('''
            class Once:
                def __init__(self): self.calls=0
                def __iter__(self):
                    self.calls += 1
                    assert self.calls == 1
                    yield from range(100)
            items=Once()
            assert run_tasks(items, lambda x:x*x, capacity=1)==[x*x for x in range(100)]
            assert items.calls==1
            assert run_tasks([], lambda x:1/0, capacity=1)==[]
            assert run_tasks([None,0,False], lambda x:x, capacity=1)==[None,0,False]
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_worker_failure_wakes_a_producer_already_waiting_on_a_full_queue(self):
        self.run_child('''
            waiting=threading.Event(); observed=[]
            class ObservedQueue(Queue):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs);observed.append(self)
                    original=self.not_full.wait
                    def wait(timeout=None):
                        if self.putting==2: waiting.set()
                        return original(timeout)
                    self.not_full.wait=wait
                def put(self,item,*args,**kwargs):
                    self.putting=item
                    return super().put(item,*args,**kwargs)
            run_tasks.__globals__['Queue']=ObservedQueue
            failure=RuntimeError('work failed')
            called=[]
            def process(item):
                called.append(item)
                assert item==0
                assert waiting.wait(3), 'producer did not enter queue wait'
                raise failure
            try: run_tasks(range(100),process,capacity=1)
            except RuntimeError as caught: assert caught is failure
            else: raise AssertionError('worker failure was swallowed')
            assert called==[0] and waiting.is_set()
            assert observed[0].unfinished_tasks==0 and observed[0].empty()
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_producer_error_before_first_item_wakes_the_empty_queue_reader(self):
        self.run_child('''
            waiting=threading.Event()
            class ObservedQueue(Queue):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    original=self.not_empty.wait
                    def wait(timeout=None):
                        waiting.set()
                        return original(timeout)
                    self.not_empty.wait=wait
            run_tasks.__globals__['Queue']=ObservedQueue
            failure=ValueError('input interrupted')
            def items():
                if False: yield None
                assert waiting.wait(3), 'reader did not enter the empty queue wait'
                raise failure
            try: run_tasks(items(),lambda x:x,capacity=1)
            except ValueError as caught: assert caught is failure
            else: raise AssertionError('producer failure was swallowed')
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_shutdown_exceptions_from_user_code_are_reported(self):
        self.run_child('''
            for origin in ('producer','worker'):
                failure=ShutDown('user code failed')
                def items():
                    if origin=='producer': raise failure
                    yield 0
                def process(item): raise failure
                try: run_tasks(items(),process,capacity=1)
                except ShutDown as caught: assert caught is failure
                else: raise AssertionError(origin+' failure was swallowed')
                assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_base_exception_is_returned_to_the_caller(self):
        self.run_child('''
            failure=KeyboardInterrupt('worker interrupted')
            def process(item): raise failure
            try: run_tasks(range(10),process,capacity=1)
            except KeyboardInterrupt as caught: assert caught is failure
            else: raise AssertionError('worker interruption was swallowed')
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_cancellation_still_waits_for_an_inflight_task(self):
        self.run_child('''
            entered=threading.Event();release=threading.Event();aborted=threading.Event()
            class ObservedQueue(Queue):
                def shutdown(self,immediate=False):
                    super().shutdown(immediate)
                    if immediate: aborted.set()
            run_tasks.__globals__['Queue']=ObservedQueue
            failure=ValueError('source failed');caught=[]
            def items():
                yield 0
                assert entered.wait(3)
                raise failure
            def process(item):
                entered.set()
                assert release.wait(5), 'running task was never released'
                return item
            def invoke():
                try: run_tasks(items(),process,capacity=1)
                except BaseException as error: caught.append(error)
            runner=threading.Thread(target=invoke)
            runner.start()
            assert aborted.wait(3), 'producer did not cancel the queue'
            assert runner.is_alive(), 'shutdown pretended to interrupt running work'
            release.set();runner.join(3)
            assert not runner.is_alive() and caught==[failure]
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_producer_error_takes_precedence_if_both_sides_fail(self):
        self.run_child('''
            entered=threading.Event();producer_failed=threading.Event()
            first=ValueError('producer');second=RuntimeError('worker')
            def items():
                yield 0
                assert entered.wait(3)
                producer_failed.set()
                raise first
            def process(item):
                entered.set()
                assert producer_failed.wait(3)
                raise second
            try: run_tasks(items(),process,capacity=1)
            except ValueError as caught: assert caught is first
            else: raise AssertionError('producer failure lost its precedence')
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_invalid_configuration_starts_no_worker(self):
        self.run_child('''
            for capacity in (0,-1,True,1.5,None):
                try: run_tasks([],lambda x:x,capacity=capacity)
                except ValueError: pass
                else: raise AssertionError('invalid capacity accepted')
            try: run_tasks([],None)
            except ValueError: pass
            else: raise AssertionError('non-callable processor accepted')
            assert not any(t.name=='bounded-task-worker' for t in threading.enumerate())
        ''')

    def test_downloadable_demo_success_and_failure_exit_status(self):
        good = subprocess.run([sys.executable,'-B',str(EXAMPLE)],capture_output=True,text=True,timeout=10)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(ast.literal_eval(good.stdout),[x*x for x in range(10)])
        bad = subprocess.run([sys.executable,'-B',str(EXAMPLE),'--fail-at','3'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('processing task 3 failed',bad.stderr)
        self.assertEqual(bad.stdout,'')
        self.assertNotIn('Exception in thread',bad.stderr)


if __name__ == '__main__':
    unittest.main()
