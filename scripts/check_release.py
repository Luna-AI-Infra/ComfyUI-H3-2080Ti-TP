"""Check distributable structure, relative Markdown links and common privacy leaks."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', 'third_party', '.venv'}
PRIVATE = [
    re.compile(r'/(?:home|mnt|media)/[A-Za-z0-9][^\s`<>"\']*'),
    re.compile(r'[A-Za-z]:[\\/]Users[\\/][^\s`<>"\']+', re.I),
    re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
]
EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b')


def main():
    problems = []
    files = []
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in SKIP for part in rel.parts) or rel.parts[:2] == ('kernels','build'):
            continue
        files.append(path)
        if path.suffix.lower() in ('.gguf','.safetensors','.key','.pem','.nsys-rep','.ncu-rep'):
            problems.append(f'{rel}: forbidden distributable file type')
        if path.suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = '\n'.join(p.extract_text() or '' for p in reader.pages) + str(reader.metadata)
        elif path.suffix.lower() in ('.png','.jpg','.mp4'):
            # Catch literal metadata paths/tokens too; rendered pixels need separate visual review.
            text = path.read_bytes().decode('utf-8', errors='ignore')
        else:
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeError:
                problems.append(f'{rel}: unrecognized binary file')
                continue
        if any(pattern.search(text) for pattern in PRIVATE):
            problems.append(f'{rel}: possible machine path/private host/secret')
        email_text = text
        if path.suffix.lower() in ('.png', '.jpg'):
            from PIL import Image
            with Image.open(path) as im:
                email_text = json.dumps(im.info, default=str)
        elif path.suffix.lower() == '.mp4':
            if not shutil.which('ffprobe'):
                raise SystemExit('ffprobe is required for the release-media metadata check')
            email_text = subprocess.check_output(['ffprobe','-v','error','-show_entries',
                                                   'format_tags:stream_tags','-of','json',str(path)], text=True)
        # Public third-party copyright attribution, not a local committer identity.
        allowed_attribution = {'wenzel.jakob@epfl.ch'}
        if any(not x.endswith('@users.noreply.github.com') and x not in allowed_attribution for x in EMAIL.findall(email_text)):
            problems.append(f'{rel}: email address requires explicit public-attribution review')
        if path.suffix == '.py':
            ast.parse(text, filename=str(rel))
        if path.suffix == '.json':
            json.loads(text)
        if path.suffix == '.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                if target.startswith(('http://','https://','#','mailto:')):
                    continue
                target = target.split('#', 1)[0]
                if target and not (path.parent / target).exists():
                    problems.append(f'{rel}: missing relative link {target}')
    for filename in ('README.md','README.zh-CN.md','LICENSE','NOTICE','AGENTS.md'):
        if not (ROOT / filename).exists():
            problems.append('Missing ' + filename)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(json.dumps(dict(files_checked=len(files),status='passed',
                          scope='Static checks; not a substitute for human privacy review or full model validation')))


if __name__ == '__main__':
    main()
