r"""
AEGIS Step 17 — Scheduling Retrain Cycles

This guide shows how to set up automatic retraining on Windows and Linux.

========================================
WINDOWS: Task Scheduler Setup
========================================

Option 1: Using PowerShell (Easiest)
------------------------------------

1. Open PowerShell as Administrator

2. Copy-paste this command:

   $TaskPath = "AEGIS"
   $TaskName = "Retrain-Daily"
   $ScriptPath = "C:\Users\Disha\step8_9_ai_engine\retrain.py"
   $PythonExe = "C:\Users\Disha\AppData\Local\Programs\Python\Python313\python.exe"
   $WorkDir = "C:\Users\Disha\step8_9_ai_engine"

   # Create task trigger: every day at 2:00 AM
   $Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM

   # Create task action: run Python script
   $Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $ScriptPath -WorkingDirectory $WorkDir

   # Create task with high privilege
   Register-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Trigger $Trigger -Action $Action -RunLevel Highest

   Write-Host "✓ Task scheduled: AEGIS\Retrain-Daily at 2:00 AM daily"

3. Verify: Task Scheduler > Task Scheduler Library > AEGIS > Retrain-Daily

4. To remove the task later:

   Unregister-ScheduledTask -TaskPath "AEGIS" -TaskName "Retrain-Daily" -Confirm:$false


Option 2: Task Scheduler GUI (More control)
--------------------------------------------

1. Open Task Scheduler
   - Press WIN + R
   - Type: taskschd.msc
   - Press Enter

2. Create task:
   - Right-click "Task Scheduler Library" → "Create Basic Task"
   - Name: "AEGIS Retrain Daily"
   - Description: "Retrains anomaly detection models using honeypot data"
   - Click "Next >"

3. Set trigger:
   - Select "Daily"
   - Set time: 2:00 AM (when system load is low)
   - Click "Next >"

4. Set action:
   - Select "Start a program"
   - Program/script: C:\Users\Disha\AppData\Local\Programs\Python\Python313\python.exe
   - Add arguments: C:\Users\Disha\step8_9_ai_engine\retrain.py
   - Start in: C:\Users\Disha\step8_9_ai_engine
   - Click "Next >"

5. Set conditions:
   - Uncheck "Wake the computer to run this task" (optional)
   - Click "Next >"

6. Finish:
   - Check "Open the Properties dialog for this task when I click Finish"
   - Click "Finish"

7. In Properties:
   - General tab: Check "Run with highest privileges"
   - Settings tab: Set "If the task fails, restart every 10 minutes"
   - Click OK

8. Verify: Open Logs to see results
   - Right-click task → View History
   - Shows run time, exit code, success/failure


========================================
LINUX / macOS: Cron Setup
========================================

Edit your crontab:

   crontab -e

Add this line to run at 2:00 AM daily:

   0 2 * * * cd /home/user/step8_9_ai_engine && /usr/bin/python3 retrain.py >> logs/cron_retrain.log 2>&1

Or run every 6 hours:

   0 0,6,12,18 * * * cd /home/user/step8_9_ai_engine && /usr/bin/python3 retrain.py >> logs/cron_retrain.log 2>&1

Verify cron is working:

   crontab -l   # List your cron jobs
   grep CRON /var/log/syslog   # Check system cron logs


========================================
TESTING RETRAIN BEFORE AUTOMATION
========================================

1. Test dry-run (no model changes):

   python retrain.py --dry-run

   Expected output:
   - Loads honeypot data
   - Trains new models
   - Validates
   - DRY RUN: Skipping model swap
   - ✓ RETRAIN CYCLE COMPLETE

2. Test with real changes (windows):

   python retrain.py

3. Check logs:

   tail -f logs/retrain.log        # See retrain progress
   tail logs/retrain_metrics.jsonl  # See metrics (each line is JSON)


========================================
MONITORING RETRAINS
========================================

Check recent retrain history:

   # Last 5 retrain events
   tail -5 logs/retrain_metrics.jsonl | python -m json.tool

   # Check if latest retrain succeeded
   tail -1 logs/retrain_metrics.jsonl | grep "true"

Monitor retrain log in real-time:

   # Windows PowerShell
   Get-Content logs/retrain.log -Wait

   # Linux/macOS
   tail -f logs/retrain.log


========================================
TROUBLESHOOTING
========================================

Task scheduled but not running?
  ✓ Check: Task Scheduler > task name > History tab
  ✓ Check: Python path is correct
  ✓ Check: Working directory is correct
  ✓ Check: "Run with highest privileges" is enabled
  ✓ Check: System is not sleeping at 2:00 AM

Retrain fails validation?
  ✓ New model scores benign traffic too high (>20% false positives)
  ✓ Solution: Adjust contamination parameter in retrain.py
  ✓ Solution: Use --min-samples to require more honeypot data

Retrain takes too long?
  ✓ Check: How many attack samples exist (run with --min-samples 100)
  ✓ Solution: Reduce data size or run less frequently
  ✓ Solution: Profile with: python -m cProfile -s cumtime retrain.py

Model not improving?
  ✓ Check: logs/retrain_metrics.jsonl - are new metrics better?
  ✓ Solution: Run retrain manually: python retrain.py --dry-run
  ✓ Solution: Collect more diverse honeypot data


========================================
EXAMPLE RETRAIN METRICS
========================================

logs/retrain_metrics.jsonl will contain events like:

{
  "timestamp": "2026-04-14T02:00:00.001234",
  "success": true,
  "old_metrics": {
    "status": "baseline"
  },
  "new_metrics": {
    "benign_samples": 2000,
    "attack_samples": 3500,
    "honeypot_new_samples": 1200
  }
}

{
  "timestamp": "2026-04-15T02:00:00.005678",
  "success": true,
  "old_metrics": {"status": "baseline"},
  "new_metrics": {
    "benign_samples": 2000,
    "attack_samples": 4700,
    "honeypot_new_samples": 2400
  }
}

Track improvement: Each retrain should increase attack_samples and honeypot_new_samples.


========================================
PRODUCTION BEST PRACTICES
========================================

1. Schedule at low-traffic time (2 AM is good)
2. Monitor retrain log for errors
3. Set up alerts if retrain fails 3x in a row
4. Keep backup models (*.backup files)
5. Test new models before deploying (already done with validation step)
6. Set --min-samples high enough to ensure quality data
7. Review retrain_metrics.jsonl weekly to confirm improvement
8. Archive old logs monthly to manage disk space

For critical systems:
  - Test retrain in staging environment first
  - Run manual retrain before schedule takes over
  - Keep the old model.backup file as fallback
  - Alert on validation failures
"""

# Auto-setup script for Windows users
if __name__ == "__main__":
    import platform

    if platform.system() == "Windows":
        print(__doc__)
        print("\nTo set up automatic daily retrains, copy-paste the PowerShell command above.")
    else:
        print(__doc__)
        print("\nTo set up automatic daily retrains, edit your crontab:")
        print("  crontab -e")
