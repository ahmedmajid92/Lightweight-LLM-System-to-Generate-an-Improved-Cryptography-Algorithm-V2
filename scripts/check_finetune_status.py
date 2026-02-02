"""Check the status of an existing fine-tuning job.

Usage:
    python scripts/check_finetune_status.py <JOB_ID>
    python scripts/check_finetune_status.py --list  # List all jobs
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def list_jobs(client: OpenAI, limit: int = 10):
    """List recent fine-tuning jobs."""
    print(f"\n📋 Recent Fine-Tuning Jobs (last {limit}):\n")
    
    jobs = client.fine_tuning.jobs.list(limit=limit)
    
    if not jobs.data:
        print("   No fine-tuning jobs found.")
        return
    
    for job in jobs.data:
        status_icon = {
            "succeeded": "✅",
            "failed": "❌",
            "cancelled": "⚠️",
            "running": "🔄",
            "queued": "⏳",
            "validating_files": "📋",
        }.get(job.status, "❓")
        
        print(f"{status_icon} {job.id}")
        print(f"   Model: {job.model}")
        print(f"   Status: {job.status}")
        if job.fine_tuned_model:
            print(f"   Fine-tuned: {job.fine_tuned_model}")
        print()


def check_job(client: OpenAI, job_id: str):
    """Check status of a specific job."""
    print(f"\n🔍 Checking job: {job_id}\n")
    
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print(f"Status: {job.status}")
    print(f"Model: {job.model}")
    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")
    if job.error:
        print(f"Error: {job.error}")
    
    # Show recent events
    print("\n📝 Recent Events:")
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
    for event in reversed(events.data):
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.created_at))
        print(f"   [{timestamp}] {event.message}")


def main():
    parser = argparse.ArgumentParser(description="Check fine-tuning status")
    parser.add_argument("job_id", nargs="?", help="Job ID to check")
    parser.add_argument("--list", action="store_true", help="List all jobs")
    args = parser.parse_args()
    
    client = get_client()
    
    if args.list or not args.job_id:
        list_jobs(client)
    else:
        check_job(client, args.job_id)


if __name__ == "__main__":
    main()
