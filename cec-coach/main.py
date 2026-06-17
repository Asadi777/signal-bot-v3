#!/usr/bin/env python3
"""CEC Coach entry point.

Usage:
    python main.py            # interactive coach
    python main.py import     # import question files from data/incoming/
    python main.py stats      # print library stats and exit
"""

import sys

from cec_coach import cli, store


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None

    if cmd == "import":
        report = store.import_incoming(apply=True)
        print(f"Imported {report['added']} new questions, skipped {report['skipped']}.")
        for f in report["files"]:
            print(f"  {f['file']}: +{f['ok']} ok, {len(f['dupes'])} dupes, "
                  f"{len(f['problems'])} problems")
            for p in f["problems"][:20]:
                print(f"      - {p}")
        return

    if cmd == "stats":
        questions, warnings = store.load_all()
        st = store.stats(questions)
        print(f"Total questions: {st['total']}")
        print(f"Need review:     {st['needs_review']}")
        print("By block:  ", st["by_block"])
        print("By section:", st["by_section"])
        print("By set:    ", st["by_set"])
        if warnings:
            print(f"\n{len(warnings)} warning(s):")
            for w in warnings:
                print("  -", w)
        return

    cli.main()


if __name__ == "__main__":
    main(sys.argv)
