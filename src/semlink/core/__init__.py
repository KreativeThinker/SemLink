def run_task(name: str, dry_run: bool = False) -> str:
    if dry_run:
        return f"[dry-run] would run task: {name}"

    return f"running task: {name}"
