"""Offline SWE integration for the vendored ReasoningBank agent."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/src'))
os.environ.setdefault('MSWEA_SILENT_STARTUP','1')
os.environ.setdefault('MSWEA_GLOBAL_CONFIG_DIR',str(ROOT/'outputs/mini_config'))
