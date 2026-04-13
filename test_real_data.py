"""
Test risk_scorer with real attack data from CSV files
"""
import pandas as pd
from risk_scorer import compute_risk

# Load real attack data
print("=" * 70)
print("TESTING RISK SCORER WITH REAL ATTACK DATA")
print("=" * 70)
print()

# Test DDoS attacks
print("1. DDoS Attack Flows (Friday-Afternoon-DDos)")
print("-" * 70)
df_ddos = pd.read_csv('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=3)

for idx in range(len(df_ddos)):
    row = df_ddos.iloc[idx].to_dict()
    anomaly_score = 0.82  # High anomaly from Step 8
    result = compute_risk(anomaly_score, row)
    
    syn = row.get(' SYN Flag Count', 0)
    psh = row.get(' PSH Flag Count', 0)
    fin = row.get('FIN Flag Count', 0)
    
    print(f"Flow {idx+1}: Flags(SYN={syn}, PSH={psh}, FIN={fin})")
    print(f"  Abnormal? YES → anomaly_score={anomaly_score}")
    print(f"  → risk_score={result['risk_score']} ({result['risk_level']})")
    print(f"  → breakdown: {result['breakdown']}")
    print()

# Test Port Scan
print("\n2. Port Scan Attack Flows (Friday-Afternoon-PortScan)")
print("-" * 70)
df_scan = pd.read_csv('data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', nrows=3)

for idx in range(len(df_scan)):
    row = df_scan.iloc[idx].to_dict()
    anomaly_score = 0.65  # Medium-high anomaly
    result = compute_risk(anomaly_score, row)
    
    syn = row.get(' SYN Flag Count', 0)
    psh = row.get(' PSH Flag Count', 0)
    fin = row.get('FIN Flag Count', 0)
    
    print(f"Flow {idx+1}: Flags(SYN={syn}, PSH={psh}, FIN={fin})")
    print(f"  Abnormal? YES → anomaly_score={anomaly_score}")
    print(f"  → risk_score={result['risk_score']} ({result['risk_level']})")
    print(f"  → breakdown: {result['breakdown']}")
    print()

# Test benign traffic
print("\n3. Benign Traffic (Friday-Morning)")
print("-" * 70)
df_benign = pd.read_csv('data/Friday-WorkingHours-Morning.pcap_ISCX.csv', nrows=3)

for idx in range(len(df_benign)):
    row = df_benign.iloc[idx].to_dict()
    anomaly_score = 0.15  # Low anomaly - normal traffic
    result = compute_risk(anomaly_score, row)
    
    syn = row.get(' SYN Flag Count', 0)
    psh = row.get(' PSH Flag Count', 0)
    fin = row.get('FIN Flag Count', 0)
    
    print(f"Flow {idx+1}: Flags(SYN={syn}, PSH={psh}, FIN={fin})")
    print(f"  Abnormal? NO → anomaly_score={anomaly_score}")
    print(f"  → risk_score={result['risk_score']} ({result['risk_level']})")
    print(f"  → breakdown: {result['breakdown']}")
    print()

print("\n✓ Real data testing complete!")
print("\nNote: Risk scores are LOW because real CSV has binary flags (0/1).")
print("Normalization limits in config (syn_flag_count_max=100) are conservative.")
print("\nTo see higher risk differentiation, adjust risk_scorer_config.json:")
print("  - Lower syn_flag_count_max from 100 → 1")
print("  - Lower pst_flag_count_max from 50 → 1")
print("  - Lower fin_flag_count_max from 50 → 1")
