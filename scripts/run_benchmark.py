from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research.cli import benchmark_main


if __name__ == "__main__":
    benchmark_main()

