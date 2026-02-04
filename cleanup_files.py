# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Cleanup script for Neurova codebase.

This script:
1. Removes all emojis from files
2. Adds proper copyright headers to files missing them
3. Ensures consistent formatting
"""

import os
import re
from pathlib import Path

# Emoji pattern - comprehensive Unicode emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U0001F000-\U0001F02F"  # Mahjong Tiles
    "\U0001F0A0-\U0001F0FF"  # Playing Cards
    "]+",
    flags=re.UNICODE
)

# Copyright headers by file extension
COPYRIGHT_HEADERS = {
    ".py": """# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

""",
    ".js": """// copyright (c) 2025 @squid consultancy group (scg)
// all rights reserved.
// licensed under the mit license.

""",
    ".ts": """// copyright (c) 2025 @squid consultancy group (scg)
// all rights reserved.
// licensed under the mit license.

""",
    ".css": """/* copyright (c) 2025 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the mit license.
 */

""",
    ".c": """/* copyright (c) 2025 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the mit license.
 */

""",
    ".cpp": """/* copyright (c) 2025 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the mit license.
 */

""",
    ".h": """/* copyright (c) 2025 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the mit license.
 */

""",
    ".hpp": """/* copyright (c) 2025 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the mit license.
 */

""",
    ".yaml": """# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

""",
    ".yml": """# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

""",
    ".toml": """# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

""",
    ".sh": """#!/bin/bash
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

""",
    ".cmake": """# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

""",
}

# Directories to skip
SKIP_DIRS = {
    '.git', '__pycache__', '.pytest_cache', 'build', 'dist',
    'htmlcov', '.egg-info', 'node_modules', '.venv', 'venv',
    'neurova.egg-info', 'site-packages'
}

# Files to skip
SKIP_FILES = {
    'LICENSE', 'MANIFEST.in', '.gitignore', '.coverage',
    'requirements.txt', 'setup.cfg'
}

# Extensions to process
PROCESS_EXTENSIONS = {'.py', '.md', '.rst', '.txt', '.yaml', '.yml', '.toml',
                      '.js', '.ts', '.css', '.c', '.cpp', '.h', '.hpp',
                      '.sh', '.cmake'}


def should_skip_path(path: Path) -> bool:
    """Check if path should be skipped."""
    # Skip directories
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    
    # Skip specific files
    if path.name in SKIP_FILES:
        return True
    
    return False


def remove_emojis(text: str) -> str:
    """Remove all emojis from text."""
    return EMOJI_PATTERN.sub('', text)


def has_copyright_header(content: str, ext: str) -> bool:
    """Check if file already has copyright header."""
    first_lines = content[:500].lower()
    return 'copyright' in first_lines and '@analytics' in first_lines


def add_copyright_header(content: str, ext: str) -> str:
    """Add copyright header to file content."""
    header = COPYRIGHT_HEADERS.get(ext, "")
    if not header:
        return content
    
    # Handle Python files with shebang
    if ext == ".py" and content.startswith("#!"):
        lines = content.split('\n', 1)
        shebang = lines[0] + '\n'
        rest = lines[1] if len(lines) > 1 else ""
        return shebang + header + rest
    
    # Handle shell scripts (shebang already in header)
    if ext == ".sh" and content.startswith("#!/"):
        # Remove existing shebang, use header's shebang
        lines = content.split('\n', 1)
        rest = lines[1] if len(lines) > 1 else ""
        return header + rest
    
    return header + content


def process_file(filepath: Path, dry_run: bool = False) -> dict:
    """Process a single file."""
    result = {
        'path': str(filepath),
        'emojis_removed': 0,
        'header_added': False,
        'modified': False,
        'error': None
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        original_content = content
        ext = filepath.suffix.lower()
        
        # Step 1: Remove emojis
        new_content = remove_emojis(content)
        emoji_count = len(content) - len(new_content)
        result['emojis_removed'] = emoji_count
        
        # Step 2: Add copyright header if missing (for code files)
        if ext in COPYRIGHT_HEADERS:
            if not has_copyright_header(new_content, ext):
                new_content = add_copyright_header(new_content, ext)
                result['header_added'] = True
        
        # Check if file was modified
        if new_content != original_content:
            result['modified'] = True
            if not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def process_directory(root_dir: Path, dry_run: bool = False) -> list:
    """Process all files in directory."""
    results = []
    
    for filepath in root_dir.rglob('*'):
        if not filepath.is_file():
            continue
        
        if should_skip_path(filepath):
            continue
        
        if filepath.suffix.lower() not in PROCESS_EXTENSIONS:
            continue
        
        result = process_file(filepath, dry_run)
        results.append(result)
    
    return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cleanup Neurova codebase')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--dir', default='.',
                        help='Directory to process (default: current)')
    args = parser.parse_args()
    
    root_dir = Path(args.dir).resolve()
    print(f"Processing: {root_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("-" * 60)
    
    results = process_directory(root_dir, args.dry_run)
    
    # Summary
    modified = [r for r in results if r['modified']]
    headers_added = [r for r in results if r['header_added']]
    emojis_removed = sum(r['emojis_removed'] for r in results)
    errors = [r for r in results if r['error']]
    
    print(f"\nSummary:")
    print(f"  Files processed: {len(results)}")
    print(f"  Files modified: {len(modified)}")
    print(f"  Headers added: {len(headers_added)}")
    print(f"  Emojis removed: {emojis_removed}")
    print(f"  Errors: {len(errors)}")
    
    if modified and args.dry_run:
        print("\nFiles that would be modified:")
        for r in modified[:20]:
            changes = []
            if r['emojis_removed'] > 0:
                changes.append(f"{r['emojis_removed']} emojis")
            if r['header_added']:
                changes.append("add header")
            print(f"  {r['path']}: {', '.join(changes)}")
        if len(modified) > 20:
            print(f"  ... and {len(modified) - 20} more files")
    
    if errors:
        print("\nErrors:")
        for r in errors[:10]:
            print(f"  {r['path']}: {r['error']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
