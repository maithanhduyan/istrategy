#!/usr/bin/env python3
"""Entry point for reasoning agent application"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.main import main

    main()
