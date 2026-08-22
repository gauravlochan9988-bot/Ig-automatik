#!/usr/bin/env python3
"""IG-AUTOMATIK - Simple entry point for batch processing.

Usage:
    python main.py                  # Process all files in 1_EINGANG
    python main.py --limit 5        # Process first 5 files
    python main.py --help           # Show help
"""

import sys
import argparse
from pathlib import Path

# Add package to path. Works whether this file sits in the project root or in
# a subfolder such as _SYSTEM/, so the entry point keeps working if it is moved.
_here = Path(__file__).resolve().parent
for _candidate in (_here, *_here.parents):
    if (_candidate / "ig_automatik").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from ig_automatik.config import Config
from ig_automatik.core import run_on_folder
from ig_automatik.utils import get_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IG-AUTOMATIK: Professional Instagram content grading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Process all files
  python main.py --limit 3        Process first 3 files
  python main.py --help           Show this help
        """
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="IG-AUTOMATIK 2.0.0"
    )

    args = parser.parse_args()

    logger = get_logger()

    print("""
    ╔════════════════════════════════════╗
    ║     IG-AUTOMATIK v2.0              ║
    ║  Instagram Content Grading Engine  ║
    ╚════════════════════════════════════╝
    """)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        cfg = Config.load()

        # Show input folder
        input_folder = Path(cfg["input_folder"])
        output_folder = Path(cfg["output_folder"])

        print(f"📂 Input:  {input_folder}")
        print(f"📂 Output: {output_folder}")
        print()

        if not input_folder.exists():
            input_folder.mkdir(parents=True, exist_ok=True)
            logger.warn(f"Created input folder: {input_folder}")

        # Run processing
        logger.info("Starting batch processing...")
        run_on_folder(cfg, batch_limit=args.limit)

        print(f"\n✅ Processing complete!")
        print(f"📁 Check {output_folder} for results")

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Processing failed", error=e)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
