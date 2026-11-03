"""One CLI for every modality/objective combination and diagnostic stage."""

import argparse
from pathlib import Path

from .config import load_config
from .objective_parser import interpret
from .orchestrator import STAGES, run, run_stage
from .utils import check_environment


def main():
    check_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=STAGES + ["run", "smoke", "project-report"])
    parser.add_argument("--config")
    parser.add_argument("--objective")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", choices=STAGES)
    parser.add_argument("--through", choices=STAGES)
    args = parser.parse_args()
    if args.command == "smoke":
        from .smoke import smoke

        smoke()
        return
    if args.command == "project-report":
        from .project_report import build_project_report

        build_project_report()
        return
    if not args.config or not args.objective:
        parser.error("Dataset config and user objective are required")
    cfg = load_config(args.config)
    objective_path = Path(args.objective)
    try:
        is_file = objective_path.is_file()
    except OSError:
        is_file = False
    if not is_file and objective_path.suffix in {'.yaml', '.yml', '.json', '.txt'}:
        raise FileNotFoundError(f'Objective file does not exist: {objective_path}')
    obj = interpret(objective_path if is_file else args.objective)
    if args.command == "run":
        run(cfg, obj, resume=args.resume, force_stage=args.force, through=args.through)
    else:
        run_stage(cfg, obj, args.command, resume=args.resume, force=args.force == args.command)


if __name__ == "__main__":
    main()
