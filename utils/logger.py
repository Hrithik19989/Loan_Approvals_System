# utils/logger.py
"""
Structural logging engine setting up unified trace outputs.
Manages console streaming and automatic rotating file logging blocks.
"""

import sys
from loguru import logger

# Wipe standard basic logger listeners
logger.remove()

# Config 1: Clean colorized stdout console stream mapping
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Config 2: Rotating execution log text matrix for persistent trace audits
logger.add(
    "logs/underwriting_execution.log",
    rotation="10 MB",
    retention="15 days",
    compression="zip",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)

def get_production_logger():
    """Returns the globally configured Loguru instance wrapper."""
    return logger
