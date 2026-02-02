"""Test the fine-tuned model with sample prompts.

Usage:
    python scripts/test_finetuned_model.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_model():
    """Get the fine-tuned model ID from .env or use default."""
    model = os.getenv("FINETUNED_MODEL")
    if not model:
        print("⚠️ FINETUNED_MODEL not found in .env, using base model")
        model = "gpt-4.1-mini"
    return model


def test_cipher_spec(client: OpenAI, model: str):
    """Test CipherSpec generation."""
    print("\n" + "=" * 50)
    print("Test 1: CipherSpec Generation")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate CipherSpec JSON for Crypto Cipher Lab v2. Output ONLY valid JSON."},
            {"role": "user", "content": "Create a CipherSpec for a Feistel cipher based on Blowfish with 128-bit keys and 16 rounds."}
        ],
        temperature=0.7,
    )
    
    result = response.choices[0].message.content
    print(f"\nResponse:\n{result}")
    
    # Try to parse as JSON
    try:
        parsed = json.loads(result)
        print("\n✅ Valid JSON")
    except:
        print("\n⚠️ Response is not valid JSON")


def test_improvement(client: OpenAI, model: str):
    """Test ImprovementPatch generation."""
    print("\n" + "=" * 50)
    print("Test 2: ImprovementPatch Generation")
    print("=" * 50)
    
    spec = {
        "name": "WeakCipher",
        "architecture": "SPN",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 6,
        "components": {"sbox": "sbox.identity", "perm": "perm.identity", "linear": "linear.identity"}
    }
    
    metrics = {"plaintext_avalanche": {"mean": 0.15}, "key_avalanche": {"mean": 0.18}}
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate ImprovementPatch JSON for Crypto Cipher Lab v2. Provide clear rationale."},
            {"role": "user", "content": f"The following cipher has poor avalanche. Suggest improvements.\nSpec: {json.dumps(spec)}\nMetrics: {json.dumps(metrics)}"}
        ],
        temperature=0.7,
    )
    
    result = response.choices[0].message.content
    print(f"\nResponse:\n{result}")


def test_code_generation(client: OpenAI, model: str):
    """Test Python code generation."""
    print("\n" + "=" * 50)
    print("Test 3: Code Generation with Comments")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write Python code for Crypto Cipher Lab. Include detailed comments."},
            {"role": "user", "content": "Write Python code to create an AES-style SPN cipher with 10 rounds, export it to a file, and run self_test()."}
        ],
        temperature=0.7,
        max_tokens=800,
    )
    
    result = response.choices[0].message.content
    print(f"\nResponse:\n{result[:1000]}...")  # Truncate for display


def test_qa(client: OpenAI, model: str):
    """Test Q&A about block ciphers."""
    print("\n" + "=" * 50)
    print("Test 4: Block Cipher Q&A")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a cryptography tutor. Be accurate and avoid unsupported claims."},
            {"role": "user", "content": "Explain the difference between SPN and Feistel architectures in 100 words."}
        ],
        temperature=0.7,
    )
    
    result = response.choices[0].message.content
    print(f"\nResponse:\n{result}")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env")
        return
    
    client = OpenAI(api_key=api_key)
    model = get_model()
    
    print("🧪 Testing Fine-Tuned Model")
    print(f"   Model: {model}")
    
    test_cipher_spec(client, model)
    test_improvement(client, model)
    test_code_generation(client, model)
    test_qa(client, model)
    
    print("\n" + "=" * 50)
    print("✅ All tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
