"""One-time demo environment setup; never installs into system Python."""
from pathlib import Path
import platform, subprocess, sys, venv
ROOT = Path(__file__).resolve().parent

def main():
    if sys.version_info[:2] != (3, 13):
        raise SystemExit('Use 64-bit Python 3.13 to match the validated environment.')
    if sys.maxsize <= 2**32:
        raise SystemExit('Install 64-bit Python.')
    env = ROOT / '.venv'
    python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    def pip(*args):
        subprocess.run([str(python), '-m', 'pip', *args], check=True, cwd=ROOT)
    pip('install', '--upgrade', 'pip')
    torch_args = ['install', 'torch==2.14.0']
    if sys.platform in ('linux', 'win32') and platform.machine().lower() in ('amd64', 'x86_64'):
        torch_args += ['--index-url', 'https://download.pytorch.org/whl/cpu']
    pip(*torch_args)
    pip('install', '-r', str(ROOT / 'requirements.txt'))
    subprocess.run([str(python), str(ROOT / 'run_demo.py'), '--check'], check=True, cwd=ROOT)
    print('Setup complete. Use the run script for your operating system.')

if __name__ == '__main__':
    main()
