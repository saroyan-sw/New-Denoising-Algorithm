"""Training entry point; Milestone 0 provides recoverable state inspection."""

import argparse

from state import status_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="show persistent study state")
    args = parser.parse_args()
    if args.status:
        print(status_text())
        return
    parser.error("training is implemented in Milestone 1; run --status for current state")


if __name__ == "__main__":
    main()
