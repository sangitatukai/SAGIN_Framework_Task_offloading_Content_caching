# quick_fix.py - Auto-fix your rl_baseline_comparison.py
import re

print("🔧 Fixing rl_baseline_comparison.py...")

# Read the file with proper encoding handling
try:
    with open('rl_baseline_comparison.py', 'r', encoding='utf-8') as f:
        content = f.read()
    print("✅ File read with UTF-8 encoding")
except UnicodeDecodeError:
    try:
        with open('rl_baseline_comparison.py', 'r', encoding='cp1252') as f:
            content = f.read()
        print("✅ File read with CP1252 encoding")
    except UnicodeDecodeError:
        with open('rl_baseline_comparison.py', 'r', encoding='latin1') as f:
            content = f.read()
        print("✅ File read with Latin1 encoding")

print("✅ File read successfully")

# Fix 1: Change imports
old_import = 'from rl_formulation_sagin import NuclearSAGINAgent'
new_import = 'from rl_formulation_sagin import HierarchicalSAGINAgent'
if old_import in content:
    content = content.replace(old_import, new_import)
    print("✅ Fixed import statement")

# Fix 2: Change agent creation
content = re.sub(
    r'rl_agent = NuclearSAGINAgent\(',
    'rl_agent = HierarchicalSAGINAgent(',
    content
)
print("✅ Fixed agent creation")

# Fix 3: Remove device_selection lines from baseline configs
lines = content.split('\n')
fixed_lines = []
for line in lines:
    if "'device_selection'" not in line:
        fixed_lines.append(line)
    else:
        print(f"✅ Removed line: {line.strip()}")

content = '\n'.join(fixed_lines)

# Fix 4: Replace device_selection references with aggregation
content = content.replace(
    "config['device_selection']",
    "config['aggregation']"
)
print("✅ Fixed device_selection references")

# Write back the fixed content with safe encoding
try:
    with open('rl_baseline_comparison.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ File written with UTF-8 encoding")
except:
    with open('rl_baseline_comparison.py', 'w', encoding='cp1252') as f:
        f.write(content)
    print("✅ File written with CP1252 encoding")

print("🎉 rl_baseline_comparison.py has been fixed!")
print("🚀 Now run: python rl_baseline_comparison.py")