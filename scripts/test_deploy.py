#!/usr/bin/env python3
"""Exercise manual deployment only with disposable sites and local bare remotes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='chase-deploy-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_CONFIG_NOSYSTEM='1', GIT_ALLOW_PROTOCOL='file')
        self.site = self.root/'site with spaces'
        self.site.mkdir()
        self.remote = self.root/'local remote.git'
        self.run_git('init', '--bare', str(self.remote))
        self.run_git('init', '-b', 'main', str(self.site))
        for key, value in [('user.name', 'Deployment fixture'), ('user.email', 'fixture@example.invalid')]:
            self.run_git('-C', str(self.site), 'config', key, value)
        self.run_git('-C', str(self.site), 'remote', 'add', 'origin', str(self.remote))
        shutil.copyfile(ROOT/'deploy.sh', self.site/'deploy.sh')
        self.payload = self.site/'payload'
        self.payload.mkdir()
        (self.payload/'index.html').write_text('published version one')
        self.public = self.site/'public'
        self.public.mkdir()
        (self.public/'old-draft.html').write_text('cached draft, never publish')
        self.staging = self.root/'temporary builds'
        self.staging.mkdir()
        self.trace = self.root/'trace'
        self.trace.mkdir()
        binary = self.root/'hugo fixture'
        binary.write_text('''#!/usr/bin/env python3
import os, pathlib, shutil, sys
args = sys.argv[1:]
assert '--minify' in args and '--buildDrafts=false' in args
assert args[args.index('--environment')+1] == 'production'
source = pathlib.Path(args[args.index('--source')+1])
dest = pathlib.Path(args[args.index('--destination')+1])
assert not dest.exists() or not list(dest.iterdir()), 'Reused output directory'
pathlib.Path(os.environ['DEPLOY_TEST_TRACE'], 'build').write_text(str(source))
if (source/'fail-build').exists(): sys.exit(31)
shutil.copytree(source/'payload', dest, dirs_exist_ok=True)
''')
        binary.chmod(0o755)
        scripts = self.site/'scripts'
        scripts.mkdir()
        (scripts/'validate_blog.py').write_text('''import os, pathlib, sys
args = sys.argv[1:]
assert all(x in args for x in ['--python-snippets', '--structured-snippets', '--strict-math'])
pathlib.Path(os.environ['DEPLOY_TEST_TRACE'], 'validated').touch()
root = pathlib.Path(args[args.index('--public')+1])
if not (root/'index.html').is_file() or list(root.rglob('*draft*')): sys.exit(32)
''')
        (scripts/'test_site_assets.py').write_text('''import os, pathlib, sys
root = pathlib.Path(sys.argv[sys.argv.index('--public')+1])
assert (root/'index.html').is_file()
pathlib.Path(os.environ['DEPLOY_TEST_TRACE'], 'assets-validated').touch()
if (root/'fail-assets').exists(): sys.exit(33)
''')
        self.env.update(HUGO_BIN=str(binary), TMPDIR=str(self.staging),
                        DEPLOY_TEST_TRACE=str(self.trace), GIT_TERMINAL_PROMPT='0')
        self.env.pop('DEPLOY_REMOTE', None)
        self.env.pop('GIT_DIR', None)
        self.env.pop('GIT_WORK_TREE', None)
        for key in ('GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL'):
            self.env.pop(key, None)
        self.env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='commit.gpgsign', GIT_CONFIG_VALUE_0='false')

    def run_git(self, *args, check=True):
        return subprocess.run(['git', *args], text=True, capture_output=True,
                              check=check, timeout=15, env=self.env)

    def publish(self, message='fixture publication', cwd=None):
        result = subprocess.run(['bash', str(self.site/'deploy.sh'), message],
            cwd=cwd or self.root, env=self.env, text=True, capture_output=True, timeout=20)
        self.assertEqual(list(self.staging.iterdir()), [], 'Temporary checkout leaked')
        self.assertEqual((self.public/'old-draft.html').read_text(), 'cached draft, never publish')
        self.assertEqual(self.run_git('-C', str(self.site), 'branch', '--show-current').stdout.strip(), 'main')
        return result

    def head(self):
        return self.run_git('--git-dir', str(self.remote), 'rev-parse', 'refs/heads/gh-pages').stdout.strip()

    def tree(self):
        return set(self.run_git('--git-dir', str(self.remote), 'ls-tree', '-r', '--name-only', 'gh-pages').stdout.splitlines())

    def reject_pushes(self):
        hook = self.remote/'hooks/pre-receive'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)

    def test_first_publish_uses_fresh_output_and_local_identity(self):
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.tree(), {'index.html', '.nojekyll'})
        self.assertEqual((self.trace/'build').read_text(), str(self.site))
        author = self.run_git('--git-dir', str(self.remote), 'log', '-1', '--format=%an', 'gh-pages').stdout.strip()
        self.assertEqual(author, 'Deployment fixture')

    def test_update_removes_old_remote_files_without_rewriting_history(self):
        (self.payload/'obsolete.html').write_text('obsolete')
        self.assertEqual(self.publish().returncode, 0)
        first = self.head()
        (self.payload/'obsolete.html').unlink()
        (self.payload/'new.html').write_text('new')
        result = self.publish('second publication')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.tree(), {'index.html', 'new.html', '.nojekyll'})
        parent = self.run_git('--git-dir', str(self.remote), 'rev-parse', 'gh-pages^').stdout.strip()
        self.assertEqual(parent, first)

    def test_identical_output_skips_commit_and_push(self):
        self.assertEqual(self.publish().returncode, 0)
        first = self.head()
        self.reject_pushes()
        result = self.publish('unchanged publication')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.head(), first)
        self.assertIn('内容未变化', result.stdout)

    def test_global_ignore_cannot_omit_validated_build_assets(self):
        ignore = self.root/'global ignore'
        ignore.write_text('*.min.css\n')
        self.env.update(GIT_CONFIG_COUNT='2', GIT_CONFIG_KEY_1='core.excludesFile', GIT_CONFIG_VALUE_1=str(ignore))
        (self.payload/'style.min.css').write_text('body {color: black}')
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('style.min.css', self.tree())

    def test_build_failure_stops_before_validation(self):
        (self.site/'fail-build').touch()
        result = self.publish()
        self.assertEqual(result.returncode, 31)
        self.assertFalse((self.trace/'validated').exists())
        self.assertNotEqual(self.run_git('--git-dir', str(self.remote), 'show-ref', '--verify', 'refs/heads/gh-pages', check=False).returncode, 0)

    def test_validation_failure_stops_before_remote_contact(self):
        (self.payload/'draft.html').write_text('draft')
        self.env['DEPLOY_REMOTE'] = str(self.root/'missing remote')
        result = self.publish()
        self.assertEqual(result.returncode, 32, result.stderr)
        self.assertTrue((self.trace/'validated').exists())
        self.assertNotIn('手动发布分支已推送', result.stdout)

    def test_push_rejection_is_not_reported_as_success(self):
        self.assertEqual(self.publish().returncode, 0)
        first = self.head()
        self.reject_pushes()
        (self.payload/'index.html').write_text('changed')
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.head(), first)
        self.assertIn('pre-receive hook declined', result.stderr)
        self.assertNotIn('手动发布分支已推送', result.stdout)

    def test_asset_failure_stops_before_remote_contact(self):
        (self.payload/'fail-assets').touch()
        self.env['DEPLOY_REMOTE'] = str(self.root/'missing remote')
        result = self.publish()
        self.assertEqual(result.returncode, 33, result.stderr)
        self.assertTrue((self.trace/'assets-validated').exists())
        self.assertNotIn('手动发布分支已推送', result.stdout)

    def test_unreachable_remote_is_not_a_missing_branch(self):
        self.env['DEPLOY_REMOTE'] = str(self.root/'missing remote')
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('手动发布分支已推送', result.stdout)

    def test_invocation_from_other_directory_preserves_other_public(self):
        other = self.root/'other project'
        (other/'public').mkdir(parents=True)
        marker = other/'public/index.html'
        marker.write_text('unrelated project')
        result = self.publish(cwd=other)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker.read_text(), 'unrelated project')

    def test_commit_message_is_literal(self):
        marker = self.root/'must-not-exist'
        message = 'publish $(touch '+str(marker)+') `literal` \\ 中文\nsecond line'
        result = self.publish(message)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
        actual = self.run_git('--git-dir', str(self.remote), 'log', '-1', '--format=%B', 'gh-pages').stdout.rstrip('\n')
        self.assertEqual(actual, message)


if __name__ == '__main__':
    unittest.main(verbosity=2)
