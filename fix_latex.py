import re
import sys
import os

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex for inline math: $...$ but not $$...$$
    # This is tricky. simpler approach: split by '$' and identify tokens?
    # Or use a robust regex.
    # Pattern: (?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)
    # Note: .*? is non-greedy.
    
    def replacer(match):
        inner = match.group(1)
        # Don't touch existing matches if they look suspicious or are empty
        if not inner.strip():
            return match.group(0)
        
        # Replace _ with \_ only if not already escaped
        # We can use split/join logic or regex replace
        # inner.replace('_', '\\_') might double escape if \\_ is there? 
        # But \\_ in markdown is backslash underscore.
        # If the user already wrote \\_, we shouldn't make it \\\_
        
        # Use regex to replace unescaped underscores
        # Lookbehind is hard for variable length, but here we scan the string.
        # Simple replace: replace all '_' with '\_' then replace '\\_' with '\_' ? No.
        
        new_inner = re.sub(r'(?<!\\)_', r'\\_', inner)
        return f"${new_inner}$"

    # Pattern explanation:
    # (?<!\$) : Not preceded by $
    # \$      : The delimiter
    # (?!\$)  : Not followed by $ (ensures it's single $)
    # (.*?)   : The content (group 1)
    # (?<!\$) : Content not ending with $
    # \$      : Closing delimiter
    # (?!\$)  : Not followed by $
    
    pattern = re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)')
    
    new_content = pattern.sub(replacer, content)
    
    if new_content != content:
        print(f"Fixed {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        print(f"No changes in {filepath}")

if __name__ == "__main__":
    files = sys.argv[1:]
    for f in files:
        fix_file(f)
