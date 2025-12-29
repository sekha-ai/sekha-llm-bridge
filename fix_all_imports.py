#!/usr/bin/env python3
import os
import re

def fix_imports_in_file(filepath):
    """Fix all ambiguous imports in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Define all the patterns to fix - ORDER MATTERS (most specific first)
    replacements = [
        # Service imports
        (r'^from services\.', 'from sekha_llm_bridge.services.'),
        (r'^import services\.', 'import sekha_llm_bridge.services.'),
        
        # Utils imports
        (r'^from utils\.', 'from sekha_llm_bridge.utils.'),
        (r'^import utils\.', 'import sekha_llm_bridge.utils.'),
        
        # Models imports
        (r'^from models\.', 'from sekha_llm_bridge.models.'),
        (r'^import models\.', 'import sekha_llm_bridge.models.'),
        
        # Config imports
        (r'^from config import', 'from sekha_llm_bridge.config import'),
        (r'^import config$', 'import sekha_llm_bridge.config'),
        
        # Workers imports
        (r'^from workers\.', 'from sekha_llm_bridge.workers.'),
        (r'^import workers\.', 'import sekha_llm_bridge.workers.'),
    ]
    
    original = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

def fix_all_files(directory):
    """Recursively fix all Python files in a directory"""
    fixed_count = 0
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_imports_in_file(filepath):
                    fixed_count += 1
    
    return fixed_count

# Fix src/ and tests/
if __name__ == "__main__":
    print("Fixing imports in src/...")
    src_fixed = fix_all_files('src/sekha_llm_bridge')
    print(f"Fixed {src_fixed} files in src/")
    
    print("\nFixing imports in tests/...")
    test_fixed = fix_all_files('tests')
    print(f"Fixed {test_fixed} files in tests/")
    
    print("\nDone! Run tests again.")
