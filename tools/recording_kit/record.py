"""Record four hand states twice and package the original images for return."""
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES = HERE / 'data' / 'frames'
EXPECTED = [f'{pose}_{r}' for r in range(2)
            for pose in ('pinch', 'right_near', 'left_pinch', 'left_near')]


def ask(prompt):
    while True:
        value = input(prompt).strip().lower().replace(' ', '')
        if re.fullmatch(r'[a-z0-9]+', value):
            return value
        print('Use letters and digits only, e.g. alex or roomday.')


def package(session, details, returncode, word_returncode=None):
    folder = FRAMES / session
    words = HERE / 'data' / 'words' / session
    counts = {name: len(list((folder / name).glob('*.jpg'))) for name in EXPECTED}
    if not any(counts.values()) and not words.exists():
        raise RuntimeError('No images or word recordings saved. Check the camera and try again.')
    poses_complete = returncode == 0 and (folder / 'meta.json').exists() and all(counts.values())
    summary = json.loads((words/'summary.json').read_text()) if (words/'summary.json').exists() else {}
    words_finished = word_returncode == 0 and summary.get('end_reason') in ('list_finished','time_budget')
    complete = bool(poses_complete and words_finished and summary.get('eligible_words',0)>0)
    details.update(session=session, completed=complete, poses_complete=bool(poses_complete),
        frames_by_segment=counts, word_summary=summary, capture_returncode=returncode,
        word_returncode=word_returncode, kit_version='words-20261010')
    output = HERE / 'SEND_THESE_FILES'
    output.mkdir(exist_ok=True)
    suffix = '' if complete else '_PARTIAL'
    archive = output / (session + suffix + '.zip')
    print('Packaging images and videos; this can take a minute...')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr(f'{session}/recording_info.json',json.dumps(details,indent=2))
        for kind, source in [('frames',folder),('words',words)]:
            if source.exists():
                for f in sorted(source.rglob('*')):
                    if f.is_file():z.write(f,f'{session}/{kind}/{session}/{f.relative_to(source).as_posix()}')
    print(f'\nSaved {sum(bool(n) for n in counts.values())}/8 poses and {summary.get("accepted_words",0)} accepted words.')
    if not complete:
        print('PARTIAL: send it anyway and mention what went wrong; completed recordings are still useful.')
    print(f'\nSEND THIS ONE FILE TO SHAUN:\n{archive}\n({archive.stat().st_size / 1e6:.0f} MB)')
    print('Nothing uploads automatically. See START_HERE.txt.')
    return archive, complete


def main():
    print('\nWebSwipe: about 10 minutes of recording: poses, then guided words.\n')
    name = ask('Your first name or consistent nickname: ')
    setting = ask('Setting label (e.g. roomday, roomnight, othercamera): ')
    camera = input('Camera number [Enter for 0; try 1 if wrong]: ').strip() or '0'
    if not camera.isdigit():
        raise ValueError('Camera number must be 0, 1, 2, ...')
    description = input('Brief setup (e.g. seated, room light, laptop camera): ').strip()
    session = f'{name}_{setting}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    print('\nClose camera apps. Read START_HERE.txt before beginning.')
    print('Keep the other hand down. Hold the requested pose for the entire REC period.')
    print('Click the capture window. R redo current pose | Q stop | N skip (makes session incomplete).')
    input('Press Enter when ready...')
    rc = subprocess.call([sys.executable, str(HERE / 'capture_frames.py'),
                          '--session', session, '--camera', camera, '--out', str(FRAMES),
                          '--profile', 'pinch', '--seconds', '20', '--rounds', '2', '--ready', '5'], cwd=HERE)
    print('\nPART 2: WORD SWIPES (up to 6 minutes 40 seconds, including practice/retries).')
    print('Use RIGHT hand to swipe; LEFT hand holds SPACE. Release SPACE at the last letter.')
    print('Two practice words first. Then work through the prompts until the timer ends.')
    choice = input('Enter to open the word recorder, or Q to package poses only: ').strip().lower()
    word_rc = None
    if choice != 'q':
        word_rc = subprocess.call([sys.executable, str(HERE/'record_words.py'),
            '--out', str(HERE/'data'/'words'/session), '--participant', name,
            '--camera', camera, '--seconds', '400'], cwd=HERE)
    package(session, dict(participant=name, setting=setting, camera_index=int(camera),
                          setup_description=description), rc, word_rc)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'\nCould not finish: {exc}\nSee START_HERE.txt. Original recordings remain in data/frames/.')
        sys.exit(1)
