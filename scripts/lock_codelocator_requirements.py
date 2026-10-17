"""Lock only the dependencies used by CodeLocator, not unrelated global packages."""
from pathlib import Path
from importlib.metadata import version,requires
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
pending=[Requirement(line).name for line in (ROOT/'requirements-codelocator.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
locked={}
while pending:
    name=pending.pop().lower().replace('_','-')
    if name in locked: continue
    locked[name]=version(name)
    for line in requires(name) or []:
        requirement=Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate({'extra':''}): pending.append(requirement.name)
(ROOT/'requirements-codelocator-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(locked.items()))+'\n',encoding='utf-8')
