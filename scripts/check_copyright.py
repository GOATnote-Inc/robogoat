#!/usr/bin/env python3
"""
Copyright header validation script
Ensures all source files have proper copyright notices
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

# Copyright pattern to match
COPYRIGHT_PATTERNS = [
    r"Copyright.*\d{4}.*GOATnote",
    r"Copyright.*GOATnote.*\d{4}",
    r"©.*\d{4}.*GOATnote",
]

# File extensions requiring copyright headers
REQUIRED_EXTENSIONS = {
    ".py": "# Copyright",
    ".cpp": "// Copyright",
    ".cu": "// Copyright",
    ".cuh": "// Copyright",
    ".h": "// Copyright",
    ".hpp": "// Copyright",
}

# Directories to skip
SKIP_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".eggs",
    "venv",
    ".venv",
}

def check_copyright_header(file_path: Path) -> Tuple[bool, str]:
    """
    Check if file has proper copyright header
    
    Returns:
        (has_copyright, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read first 20 lines
            content = ''.join(f.readlines()[:20])
    except (UnicodeDecodeError, PermissionError):
        return True, "Binary or unreadable file, skipping"
    
    # Check for copyright pattern
    for pattern in COPYRIGHT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True, "Copyright header found"
    
    return False, "Missing copyright header"

def main(files: List[str] = None) -> int:
    """
    Main entry point
    
    Args:
        files: List of files to check. If None, check all files in repo.
    
    Returns:
        0 if all files have copyright, 1 otherwise
    """
    if files:
        file_paths = [Path(f) for f in files]
    else:
        # Scan all files in repository
        file_paths = []
        for root, dirs, filenames in Path(".").walk():
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            
            for filename in filenames:
                file_path = root / filename
                if file_path.suffix in REQUIRED_EXTENSIONS:
                    file_paths.append(file_path)
    
    missing_copyright = []
    
    for file_path in file_paths:
        # Skip if extension doesn't require copyright
        if file_path.suffix not in REQUIRED_EXTENSIONS:
            continue
        
        # Skip if in excluded directory
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        
        has_copyright, message = check_copyright_header(file_path)
        
        if not has_copyright:
            missing_copyright.append(file_path)
    
    if missing_copyright:
        print("❌ Files missing copyright headers:")
        for file_path in missing_copyright:
            print(f"  - {file_path}")
        print()
        print("Add copyright header like:")
        print("  # Copyright (c) 2025 GOATnote Inc. All rights reserved.")
        print("  # Licensed under the Apache License, Version 2.0")
        return 1
    
    print(f"✅ All {len(file_paths)} source files have copyright headers")
    return 0

if __name__ == "__main__":
    # Get files from command line arguments
    files = sys.argv[1:] if len(sys.argv) > 1 else None
    sys.exit(main(files))

