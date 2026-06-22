# Test DISEASE_DB structure
DISEASE_DB = {
    'Healthy-FIsh': {'emoji': '✅', 'gejala': 'Tidak ada gejala', 'solusi': ['Sol1', 'Sol2']},
    'Bacterial Aeromanas Disease': {'emoji': '🦠', 'gejala': 'Gejala1', 'solusi': ['A', 'B']},
    'Streptococus': {'emoji': '⚪', 'gejala': 'Gejala2', 'solusi': ['C', 'D']},
    'Tilapia Lake Virus': {'emoji': '🦠', 'gejala': 'Gejala3', 'solusi': ['E', 'F']}
}

MODEL_CLASSES = {0: 'Bacterial Aeromanas Disease', 1: 'Healthy-FIsh', 2: 'Streptococus', 3: 'Tilapia Lake Virus'}

print("Checking DISEASE_DB alignment with Model Classes:")
for idx, class_name in MODEL_CLASSES.items():
    if class_name in DISEASE_DB:
        info = DISEASE_DB[class_name]
        print("✓ %s: Found (gejala:%d chars, solusi:%d items)" % (class_name, len(info.get('gejala','')), len(info.get('solusi',[]))))
    else:
        print("✗ %s: NOT FOUND" % class_name)
