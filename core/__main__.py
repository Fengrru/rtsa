"""CLI entry point for rtsa."""
import sys, os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cli

def main():
    cli.main()

if __name__ == "__main__":
    main()
