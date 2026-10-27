#!/usr/bin/env python3
"""Run the published progress examples against one-shot streams and failures."""
import contextlib
import io
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / 'content/posts/python/tqdm/index.md'
BLOCKS = re.findall(r'```python\n(.*?)\n```', ARTICLE.read_text(), re.S)


def examples():
    namespace = {'__name__': 'article_progress_examples'}
    for index, source in enumerate(BLOCKS):
        exec(compile(source, f'{ARTICLE}:python-block-{index}', 'exec'), namespace)
    return namespace


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressExamples(unittest.TestCase):
    def setUp(self):
        self.namespace = examples()

    def test_one_shot_batches_are_processed_once(self):
        visits = []

        def batches():
            for batch in ([1, 2], [], (3, 4, 5)):
                visits.append(batch)
                yield iter(batch)

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.namespace['process_batches'](batches()), 5)
        self.assertEqual(visits, [[1, 2], [], (3, 4, 5)])

    def test_known_total_counts_records_including_empty_batches(self):
        with contextlib.redirect_stderr(io.StringIO()):
            process = self.namespace['process_batches']
            self.assertEqual(process(iter([[1, 2], [], [3, 4, 5]]), total=5), 5)
            self.assertEqual(process(iter(()), total=0), 0)
            self.assertEqual(process(iter(())), 0)
            for total in (0, 3, 6):
                with self.subTest(total=total), self.assertRaisesRegex(ValueError, 'expected .* records, processed 5'):
                    process(iter([[1, 2], [], [3, 4, 5]]), total=total)

    def test_invalid_total_does_not_consume_input(self):
        for total in (True, False, -1, 1.5, float('nan'), float('inf'), '5'):
            iterator = iter([[1, 2], [3]])
            with self.subTest(total=total), self.assertRaises(ValueError):
                self.namespace['process_batches'](iterator, total=total)
            self.assertEqual(next(iterator), [1, 2])

    def test_non_tty_count_is_independent_of_disabled_display(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(self.namespace['process_batches'](iter([[1], [2, 3]]), total=3), 3)
        self.assertEqual(stream.getvalue(), '')

    def test_failure_closes_bar_and_preserves_exception(self):
        bars = []
        original = self.namespace['tqdm']
        stream = Terminal()

        def tracked(*args, **kwargs):
            bar = original(*args, **kwargs, file=stream)
            bars.append(bar)
            return bar

        error = RuntimeError('batch processing failed')

        class BrokenRecord:
            def __mul__(self, other):
                raise error

        self.namespace['tqdm'] = tracked
        with self.assertRaises(RuntimeError) as caught:
            self.namespace['process_batches'](iter([[1], [BrokenRecord()], [3]]), total=3)
        self.assertIs(caught.exception, error)
        self.assertEqual(bars[0].n, 1)
        self.assertTrue(bars[0].disable, 'The context must close the bar on exception')
        self.assertNotIn('100%', stream.getvalue())

    def test_generator_failure_is_propagated_without_false_completion(self):
        def batches():
            yield [1, 2]
            raise OSError('input stream interrupted')

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(OSError, 'input stream interrupted'):
            self.namespace['process_batches'](batches(), total=5)

    def test_byte_scan_handles_empty_and_partial_final_chunks(self):
        scan = self.namespace['scan_bytes']
        with tempfile.TemporaryDirectory(prefix='chase-tqdm-example-') as folder:
            path = Path(folder) / 'bytes.bin'
            for size in (0, 1, 4097):
                payload = bytes(index % 251 for index in range(size))
                path.write_bytes(payload)
                for chunk_size in (1, 3, 1024):
                    with self.subTest(size=size, chunk_size=chunk_size), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(scan(path, chunk_size), size)
                    self.assertEqual(path.read_bytes(), payload)
            for invalid in (True, False, 0, -1, 1.5, '2', None):
                with self.subTest(chunk_size=invalid), self.assertRaises(ValueError):
                    scan(path, invalid)

    def test_total_check_survives_optimized_python(self):
        script = '\n'.join(BLOCKS) + '\nprocess_batches(iter([[1, 2], [3]]), total=4)\n'
        with tempfile.TemporaryDirectory(prefix='chase-tqdm-optimized-') as folder:
            path = Path(folder) / 'progress.py'
            path.write_text(script)
            run = subprocess.run([sys.executable, '-B', '-O', str(path)], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('expected 4 records, processed 3', run.stderr)

    def test_optimized_demo_still_performs_the_work(self):
        namespace = {'__name__': '__main__'}
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(BLOCKS[0], str(ARTICLE), 'exec', optimize=2), namespace)
        self.assertEqual(namespace['item_results'], [1, 4, 9])
        self.assertEqual(namespace['empty_results'], [])
        self.assertEqual(namespace['known_count'], 5)
        self.assertEqual(namespace['streamed_count'], 5)
        self.assertEqual(namespace['empty_count'], 0)
        self.assertEqual(len(namespace['nested_results']), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
