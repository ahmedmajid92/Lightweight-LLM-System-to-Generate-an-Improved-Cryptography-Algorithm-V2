"""OpenAI Fine-Tuning Script for Block Cipher LLM

This script handles the complete fine-tuning workflow:
1. Validates the dataset format
2. Uploads training and validation files
3. Creates a fine-tuning job
4. Monitors progress and waits for completion
5. Updates .env with the new model ID

Usage:
    python scripts/finetune_openai.py

Prerequisites:
    - Set OPENAI_API_KEY in your .env file
    - Prepare train.jsonl and valid.jsonl in data/sft/

Author: PhD Thesis - Block Cipher Generator
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Organization ID is automatically determined from the API key
# Project-scoped keys (sk-proj-...) are tied to a specific project

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sft"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"
ENV_FILE = PROJECT_ROOT / ".env"

# Fine-tuning parameters
BASE_MODEL = "gpt-4.1-mini-2025-04-14"  # or gpt-4o-mini-2024-07-18
SUFFIX = "cipher-lab"  # Model suffix for identification
N_EPOCHS = 3  # Number of training epochs (auto if None)


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please set your OpenAI API key in the .env file")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def validate_jsonl(filepath: Path) -> tuple[bool, int, list[str]]:
    """Validate JSONL format and return stats.
    
    Returns:
        (is_valid, num_examples, list_of_errors)
    """
    errors = []
    count = 0
    
    if not filepath.exists():
        return False, 0, [f"File not found: {filepath}"]
    
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            count += 1
            try:
                data = json.loads(line)
                
                # Check required structure
                if "messages" not in data:
                    errors.append(f"Line {i}: Missing 'messages' key")
                    continue
                
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    errors.append(f"Line {i}: 'messages' must be list with at least 2 items")
                    continue
                
                # Check message structure
                roles_seen = set()
                for j, msg in enumerate(messages):
                    if "role" not in msg or "content" not in msg:
                        errors.append(f"Line {i}, msg {j}: Missing 'role' or 'content'")
                    else:
                        roles_seen.add(msg["role"])
                
                if "assistant" not in roles_seen:
                    errors.append(f"Line {i}: Missing assistant response")
                    
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
    
    return len(errors) == 0, count, errors


def upload_file(client: OpenAI, filepath: Path, purpose: str = "fine-tune") -> str:
    """Upload a file to OpenAI and return the file ID."""
    print(f"📤 Uploading {filepath.name}...")
    
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    
    file_id = response.id
    print(f"   ✓ Uploaded: {file_id}")
    return file_id


def create_fine_tuning_job(
    client: OpenAI,
    training_file: str,
    validation_file: Optional[str] = None,
    model: str = BASE_MODEL,
    suffix: str = SUFFIX,
    n_epochs: Optional[int] = N_EPOCHS,
) -> str:
    """Create a fine-tuning job and return the job ID."""
    print(f"\n🚀 Creating fine-tuning job...")
    print(f"   Base model: {model}")
    print(f"   Suffix: {suffix}")
    if n_epochs:
        print(f"   Epochs: {n_epochs}")
    
    params = {
        "training_file": training_file,
        "model": model,
        "suffix": suffix,
    }
    
    if validation_file:
        params["validation_file"] = validation_file
    
    if n_epochs:
        params["hyperparameters"] = {"n_epochs": n_epochs}
    
    job = client.fine_tuning.jobs.create(**params)
    
    print(f"   ✓ Job created: {job.id}")
    print(f"   Status: {job.status}")
    
    return job.id


def monitor_job(client: OpenAI, job_id: str, poll_interval: int = 60) -> str:
    """Monitor a fine-tuning job until completion.
    
    Returns:
        The fine-tuned model ID on success
    """
    print(f"\n⏳ Monitoring job {job_id}...")
    print("   (This may take 10-30 minutes depending on dataset size)")
    print()
    
    last_event_id = None
    
    while True:
        # Get job status
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        # Print recent events
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
        for event in reversed(events.data):
            if last_event_id is None or event.id > last_event_id:
                timestamp = time.strftime("%H:%M:%S", time.localtime(event.created_at))
                print(f"   [{timestamp}] {event.message}")
                last_event_id = event.id
        
        # Check status
        if job.status == "succeeded":
            print(f"\n✅ Fine-tuning complete!")
            print(f"   Model ID: {job.fine_tuned_model}")
            return job.fine_tuned_model
        
        elif job.status == "failed":
            print(f"\n❌ Fine-tuning failed!")
            print(f"   Error: {job.error}")
            sys.exit(1)
        
        elif job.status == "cancelled":
            print(f"\n⚠️ Fine-tuning was cancelled")
            sys.exit(1)
        
        # Wait and poll again
        time.sleep(poll_interval)


def update_env_file(model_id: str):
    """Add the fine-tuned model ID to .env file."""
    print(f"\n📝 Updating .env file...")
    
    env_content = ""
    if ENV_FILE.exists():
        env_content = ENV_FILE.read_text()
    
    # Check if FINETUNED_MODEL already exists
    if "FINETUNED_MODEL=" in env_content:
        # Replace existing line
        lines = env_content.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("FINETUNED_MODEL="):
                new_lines.append(f"FINETUNED_MODEL={model_id}")
            else:
                new_lines.append(line)
        env_content = "\n".join(new_lines)
    else:
        # Add new line
        if not env_content.endswith("\n"):
            env_content += "\n"
        env_content += f"\n# Fine-tuned cipher model (generated {time.strftime('%Y-%m-%d')})\n"
        env_content += f"FINETUNED_MODEL={model_id}\n"
    
    ENV_FILE.write_text(env_content)
    print(f"   ✓ Added FINETUNED_MODEL={model_id}")


def main():
    """Main fine-tuning workflow."""
    print("=" * 60)
    print("🔐 Block Cipher LLM Fine-Tuning")
    print("=" * 60)
    
    # Step 1: Load environment
    client = load_environment()
    print("✓ OpenAI client initialized")
    
    # Step 2: Validate datasets
    print("\n📋 Validating datasets...")
    
    train_valid, train_count, train_errors = validate_jsonl(TRAIN_FILE)
    if not train_valid:
        print(f"❌ Training file validation failed:")
        for err in train_errors[:5]:
            print(f"   - {err}")
        sys.exit(1)
    print(f"   ✓ train.jsonl: {train_count} examples")
    
    valid_valid, valid_count, valid_errors = validate_jsonl(VALID_FILE)
    if not valid_valid:
        print(f"❌ Validation file validation failed:")
        for err in valid_errors[:5]:
            print(f"   - {err}")
        sys.exit(1)
    print(f"   ✓ valid.jsonl: {valid_count} examples")
    
    # Step 3: Upload files
    print("\n📤 Uploading files to OpenAI...")
    train_file_id = upload_file(client, TRAIN_FILE)
    valid_file_id = upload_file(client, VALID_FILE)
    
    # Step 4: Create fine-tuning job
    job_id = create_fine_tuning_job(
        client,
        training_file=train_file_id,
        validation_file=valid_file_id,
        model=BASE_MODEL,
        suffix=SUFFIX,
        n_epochs=N_EPOCHS,
    )
    
    # Step 5: Monitor until completion
    model_id = monitor_job(client, job_id)
    
    # Step 6: Update .env with new model
    update_env_file(model_id)
    
    print("\n" + "=" * 60)
    print("🎉 Fine-tuning complete!")
    print("=" * 60)
    print(f"\nYour fine-tuned model: {model_id}")
    print("\nTo use it in your app, the model ID has been added to .env")
    print("Update your code to use os.getenv('FINETUNED_MODEL') for API calls")


if __name__ == "__main__":
    main()
