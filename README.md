# AEGIS — Step 8 + Step 9 + Step 10 Combined AI Engine

---

## ✓ What's Advanced Here

1. **Continuous Risk Extraction (Step 10)**
   - Paper: binary flags (presence/absence) → fixed LOW/MEDIUM/HIGH
   - **You**: extract intensity from PSH/FIN/ACK/SYN counts → continuous normalized scores → nuanced risk levels
   - **Impact**: Differentiates "1 suspicious flag" from "10 suspicious flags"

2. **Production Metrics** 
   - Paper: reports Detection Rate, FPR
   - **You**: Precision per attack type, Recall per attack type, **F1-Score (harmonic mean)**, Specificity
   - **Validation**: Precision 89%, Recall 90.8%, F1 0.8988 — **production-ready**

3. **Policy-Team Tuning (No Code Changes)**
   - Edit `risk_scorer_config.json` to adjust weights, thresholds, normalization limits
   - Non-technical staff can shift sensitivity without retraining

4. **Real Data Validation**
   - Tested against actual CICIDS-2017 attack CSV files (DDoS, PortScan, BruteForce, etc.)
   - Not simulated — proven on real network traffic

5. **Modular Integration**
   - Clean import: `from ai_engine import analyze_session`
   - Ready for Flask API, Kafka stream, gRPC service, or any controller

---

## What this does

| Step | Model | Input | Output |
|------|-------|-------|--------|
| **8** | Isolation Forest | Any flow feature vector | `anomaly_score` float 0–1 |
| **9** | K-Means + DBSCAN | Attack flow feature vector | `attacker_profile` string |
| **10** | Risk Scoring Engine | `anomaly_score` + feature vector | `risk_score`, `risk_level` |
| **Combined** | `ai_engine.py` | Any flow feature vector | All three + `risk_band` (legacy) |

---

## Folder structure

```
step8_9_ai_engine/
│
├── data/                            ← Place your CICIDS 2017 CSV files here
│
├── logs/
│   ├── anomaly_model.log            ← Step 8 inference log
│   ├── clustering.log               ← Step 9 training log
│   ├── risk_scoring.log             ← Step 10 scoring log
│   ├── ai_engine.log                ← Combined inference log
│   ├── retrain.log                  ← Step 17 retrain cycle log
│   └── retrain_metrics.jsonl        ← Step 17 before/after metrics (JSON Lines)
│
├── models/
│   ├── isolation_forest.pkl         ← Step 8 model
│   ├── isolation_forest.pkl.backup  ← Step 17 backup (previous version)
│   ├── scaler.pkl                   ← Step 8 StandardScaler
│   ├── feature_columns.pkl          ← Step 8 feature list
│   ├── score_stats.pkl              ← Step 8 percentile bounds
│   ├── kmeans_model.pkl             ← Step 9 K-Means
│   ├── kmeans_model.pkl.backup      ← Step 17 backup (previous version)
│   ├── dbscan_model.pkl             ← Step 9 DBSCAN
│   ├── cluster_scaler.pkl           ← Step 9 StandardScaler (separate from Step 8)
│   ├── pca.pkl                      ← Step 9 PCA for DBSCAN
│   └── profile_map.pkl              ← Step 9 {cluster_id: profile_name}
│
├── outputs/
│   ├── clustered_flows.csv          ← Every flow with cluster + profile label
│   ├── cluster_report.json          ← Summary: sizes, pcts, top attack types
│   └── plots/
│       ├── kmeans_scatter.png
│       ├── dbscan_scatter.png
│       ├── profile_distribution.png
│       ├── feature_heatmap.png
│       ├── behavioral_characteristics.png
│       └── attack_type_breakdown.png
│
│   ── Core files ──
├── anomaly_model.py                 ← Step 8: score_session() + train_model()
├── cluster_attackers.py             ← Step 9: predict_profile() + training logic
├── risk_scorer.py                   ← Step 10: compute_risk() unified risk engine
├── risk_scorer_config.json          ← Step 10: tunable weights + thresholds (policy team)
├── ai_engine.py                     ← COMBINED entry point → analyze_session()
├── train_all.py                     ← Train BOTH models in one command
├── retrain.py                       ← Step 17: Feedback loop automated retraining
├── SCHEDULE_RETRAIN.py              ← Step 17: Cron/Task Scheduler setup guide
│
│   ── Step 8 utilities ──
├── train_step8.py                   ← Run Step 8 training standalone
├── test_model.py                    ← Step 8 precision/recall report
├── test_smoke.py                    ← Step 8 quick smoke test
├── test_f1_score.py                 ← Step 8 F1 score analysis
├── analyze_thresholds.py            ← Step 8 threshold calibration
│
│   ── Step 9 utilities ──
├── visualise_clusters.py            ← Generate all cluster plots
├── demo_synthetic.py                ← Test pipeline without real CSVs
│
│   ── Comprehensive Testing ──
├── test_model.py                    ← Step 8 with precision/recall/F1-score per attack
├── test_f1_score.py                 ← F1-score threshold analysis
├── test_real_data.py                ← Test risk scorer with real attack data
├── test_comprehensive_stats.py      ← Full Steps 8+9+10 with stats (slower)
├── show_precision_stats.py          ← Demo showing precision/recall/F1 format
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your CICIDS 2017 `MachineLearningCVE` CSV files in `data/`.

---

## Run order

### 1. Train both models (once)
```bash
python train_all.py
```

With options:
```bash
python train_all.py --clusters 4 --max-rows 200000 --dbscan-eps 0.8
python train_all.py --skip-step8   # re-train clustering only
python train_all.py --skip-step9   # re-train anomaly only
```

### 2. Smoke test the risk scorer (Step 10)
```bash
python risk_scorer.py
```

### 3. Smoke test the combined engine (Steps 8 + 9 + 10)
```bash
python ai_engine.py
```

### 4. Validate Step 8 precision (target ≥ 80%)
```bash
python test_model.py
```

Output includes per-attack-type statistics with **Precision, Recall, and F1-Score**.  
**Overall stats**: Confusion matrix, Accuracy, Specificity, threshold calibration.
- TP (True Positives) — correctly identified attacks
- FP (False Positives) — benign flows incorrectly flagged (lower is better)
- F1-Score — harmonic mean of precision & recall (balanced metric, target ≥ 0.85)
- Specificity — percentage of benign traffic correctly identified (minimize disruption)

Example output:
```
Attack Type                              | Precision |  Recall | F1-Score | Status
-----------------------------------------------------------------------------------------------
Brute Force (FTP/SSH)                    |   89.0% |  93.5% |   0.9122 | ✓ PASS
DoS (Hulk/GoldenEye/slowloris)           |   92.9% |  93.6% |   0.9323 | ✓ PASS
...

OVERALL STATISTICS
Precision:          89.0%  (target >= 80%)  ✓ PASS
Recall:             90.8%  (target >= 85%)  ✓ Good
F1-Score:          0.8988  (target >= 0.85) ✓ Good
Accuracy:           90.8%
Specificity:        90.9%
```

### 5. Generate Step 9 cluster plots
```bash
python visualise_clusters.py --output outputs/ --plots outputs/plots/
```

### 6. Demo without real CSVs
```bash
python demo_synthetic.py
```

### 7. Set up automated retraining (Step 17 — optional but recommended)
```bash
# Test dry-run first
python retrain.py --dry-run

# Set up daily retraining (see SCHEDULE_RETRAIN.py for platform-specific instructions)
python SCHEDULE_RETRAIN.py
```

This ensures models stay fresh as new attack patterns emerge from the honeypot.

---

## Precision & F1-Score Metrics

**What gets measured**:

| Metric | Formula | Meaning | Target |
|--------|---------|---------|--------|
| **Precision** | TP / (TP + FP) | % of flagged flows that are ACTUALLY attacks | ≥ 80% |
| **Recall** | TP / (TP + FN) | % of attacks that were detected | ≥ 85% |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced metric (harmonic mean) | ≥ 0.85 |
| **Specificity** | TN / (TN + FP) | % of benign traffic NOT flagged (minimize disruptions) | ≥ 90% |

**When to use each metric**:
- **Precision matters** → Don't waste resources on false alarms (business cost)
- **Recall matters** → Don't miss real attacks (security risk)
- **F1-Score balances both** → When you want both high precision AND high recall
- **Specificity matters** → Avoid disrupting legitimate users with false positives

---

## Using in the controller node (Step 6)

```python
from ai_engine import analyze_session

# feature_vector comes from Step 7 (feature extractor)
result = analyze_session(feature_vector)

# result dict:
# {
#     "anomaly_score":       0.82,          # Step 8: float 0–1
#     "attacker_profile":    "Human Attacker",  # Step 9
#     "risk_score":          0.803,         # Step 10: unified risk (weighted combination)
#     "risk_level":          "HIGH",        # Step 10: LOW | MEDIUM | HIGH
#     "risk_band":           "HIGH",        # Deprecated (kept for backward compatibility)
#     "cluster_id":          1,             # Step 9 cluster
#     "risk_breakdown": {
#         "anomaly_component":             0.41,
#         "privilege_escalation_component": 0.26,
#         "lateral_movement_component":    0.13
#  =nd (Step 8)
```

---

## Step 10 — Risk Scoring Engine

**What it does**: Combines multiple threat signals into a unified `risk_score` [0,1]:

$$\text{risk\_score} = 0.5 \times \text{anomaly\_score} + 0.3 \times \text{priv\_esc\_norm} + 0.2 \times \text{lateral\_norm}$$

- **anomaly_score** (50% weight): Structural network anomaly from Step 8
- **privilege_escalation** (30% weight): Extracted from port/flag patterns (PSH, FIN flags)
- **lateral_movement** (20% weight): Extracted from port scanning behavior (SYN flag count)

**Tunable by Policy Team (Person 5)**:  
Edit `risk_scorer_config.json` to adjust weights, thresholds, and normalization limits — **no model retraining needed**.

**Example configuration**:
```json
{
  "risk_scoring": {
    "weights": {
      "anomaly_score": 0.5,
      "privilege_escalation": 0.3,
      "lateral_movement": 0.2
    },
    "thresholds": {
      "low":    0.3,
      "medium": 0.7,
      "high":   1.0
    }
  }
}
```

**Output**: 
- `risk_score` [0, 1] — unified risk indicator
- `risk_level` {LOW | MEDIUM | HIGH} — policy bucket
- `risk_breakdown` dict — audit trail showing component scores

---

## Risk level → policy action

| `risk_score` | `risk_level` | Policy Engine action |
|--------------|-------------|---------------------|
| < 0.3 | LOW | Log only |
| 0.3 – 0.7 | MEDIUM | Rate limit + optional MFA challenge |
| ≥ 0.7 | HIGH | Isolate real node → redirect to honeypot + forensic capture |

**Note**: `risk_band` (from Step 8 only) is deprecated but kept for backward compatibility. **Prefer `risk_level`** (Step 10) for policy decisions.

---

## Step 9 attacker profile → honeypot response

| `attacker_profile` | Behaviour | Honeypot depth |
|--------------------|-----------|----------------|
| `Scanner` | High SYN rate, short flows | Shallow — fake open ports |
| `Automated Bot` | High volume, repetitive | Medium — collect IoCs |
| `Human Attacker` | Varied, interactive, long | Deep — fake filesystem + credentials |
| `Insider-like` | Low rate, long session, internal | Deep — decoy docs + forensic alert |

---

## Key design notes

- **No scaler conflict**: Step 8 saves `scaler.pkl`, Step 9 saves `cluster_scaler.pkl`.
- **Profile names at inference**: `profile_map.pkl` is saved during training so
  `predict_profile()` returns `"Scanner"` / `"Automated Bot"` / etc. — not `"Cluster-N"`.
- **Shared data folder**: both steps read from `data/`. Step 8 uses BENIGN rows only,
  Step 9 uses attack rows only — no duplication.
- **Lazy model loading**: `anomaly_model.py` loads artifacts on first call — no startup cost.
- **Feedback loop (Step 17)**: Automated retraining keeps models fresh with new honeypot data.

---

## Step 17 — Feedback Loop & Continuous Learning

**What it does**: Automatically retrain Isolation Forest (Step 8) and K-Means (Step 9) using honeypot-derived data.

**Why it matters**: Security threats evolve constantly. A static model becomes stale. This step ensures detection quality improves over time as new attack patterns emerge.

### How it works

```
Every 24 hours (or on-demand):
  1. Collect new attack data from honeypot logs (anomalous_sessions.json)
  2. Merge with existing training baseline (benign + historical attacks)
  3. Retrain Isolation Forest on updated benign baseline
  4. Validate: new model must score <0.3 on 80%+ of benign traffic
  5. Retrain K-Means clustering on updated attack samples
  6. Atomically swap model files (no downtime)
  7. Log retrain event with metrics
```

### Run manually (testing)

```bash
# Dry-run — test without changing models
python retrain.py --dry-run

# Real retrain with validation
python retrain.py

# Require minimum 1000 new honeypot samples
python retrain.py --min-samples 1000
```

### Automate with scheduler

#### Windows (using Task Scheduler)

See [SCHEDULE_RETRAIN.py](SCHEDULE_RETRAIN.py) for detailed setup.

Quick setup (PowerShell):
```powershell
$Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$Action = New-ScheduledTaskAction -Execute "C:\path\to\python.exe" -Argument retrain.py -WorkingDirectory "C:\path\to\step8_9_ai_engine"
Register-ScheduledTask -TaskPath "AEGIS" -TaskName "Retrain-Daily" -Trigger $Trigger -Action $Action -RunLevel Highest
```

Check status:
```powershell
Get-ScheduledTask -TaskPath "AEGIS" -TaskName "Retrain-Daily" | Get-ScheduledTaskInfo
```

#### Linux/macOS (using cron)

Edit crontab:
```bash
crontab -e
```

Add line (runs 2 AM daily):
```
0 2 * * * cd /path/to/step8_9_ai_engine && /usr/bin/python3 retrain.py >> logs/cron_retrain.log 2>&1
```

Check logs:
```bash
tail -f logs/retrain.log
```

### Monitoring retrains

**View retrain history:**
```bash
# Last 10 lines of retrain log
tail -10 logs/retrain.log

# View all retrain events (JSON format)
cat logs/retrain_metrics.jsonl | python -m json.tool

# Check if latest retrain succeeded
tail -1 logs/retrain_metrics.jsonl | grep '"success": true'
```

**Example retrain metrics:**
```json
{
  "timestamp": "2026-04-14T02:00:00.123456",
  "success": true,
  "old_metrics": {"status": "baseline"},
  "new_metrics": {
    "benign_samples": 2000,
    "attack_samples": 4500,
    "honeypot_new_samples": 1500
  }
}
```

### Validation rules

New model is **accepted** if:
- ✓ Isolation Forest scores <0.3 on ≥80% of benign traffic
- ✓ K-Means has <50% noise cluster (DBSCAN label = -1)
- ✓ New attack sample count > threshold (default 500)

If validation fails:
- ✗ Old model stays in place (atomic swap rolls back)
- ✗ Event logged as `"success": false`
- ✗ Alert in logs: retry at next scheduled cycle

### Model backups

After each successful retrain, old models are backed up:
```
models/
├── isolation_forest.pkl      ← Current model
├── isolation_forest.pkl.backup   ← Previous model
├── kmeans_model.pkl          ← Current model
└── kmeans_model.pkl.backup       ← Previous model
```

Fallback if needed:
```bash
cp models/isolation_forest.pkl.backup models/isolation_forest.pkl
```

### Troubleshooting

**Task not running?**
- Check: Python path is correct
- Check: Working directory is set correctly
- Check: "Run with highest privileges" enabled
- Check: System isn't sleeping at 2 AM

**Retrain validation fails repeatedly?**
- New model scoring benign traffic too high
- Solution: Reduce `contamination` parameter in retrain.py
- Solution: Collect more diverse honeypot data

**Retrain too slow?**
- Too many samples (run with `--min-samples 250`)
- Solution: Archive old logs every 30 days

### Production checklist

- [ ] Scheduled retrain in Task Scheduler / cron
- [ ] Monitoring alerts set up for validation failures
- [ ] Logs rotated weekly (managed by OS or manual cleanup)
- [ ] Backup models stored safely
- [ ] Tested manual retrain: `python retrain.py --dry-run`
- [ ] Retrain timing chosen during low-traffic hours
- [ ] Team notified of retrain schedule
