"""Audited v2 entry point; original preserved in archive/pre_revision_v2_scripts."""
import sys
from pathlib import Path
from run_revision import main
if __name__ == '__main__':
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(Path(__file__).resolve().parent.parent/'results'/'revision_v2')]
    if '--family' not in sys.argv:
        sys.argv += ['--family', 'gsa,gsa_cv']
    main()
