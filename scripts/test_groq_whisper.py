"""
Test script for Groq Whisper API integration.

Run this to verify that Groq Whisper is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_whisper_client():
    """Test the GroqWhisperClient directly."""
    print("=" * 60)
    print("Testing Groq Whisper Client")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file!")
        print("   Please add your Groq API key to .env")
        print("   Get one at: https://console.groq.com")
        return False
    
    print(f"✅ GROQ_API_KEY found: {api_key[:10]}...")
    
    try:
        from src.stt import GroqWhisperClient
        print("✅ GroqWhisperClient imported successfully")
        
        # Initialize client
        client = GroqWhisperClient()
        print("✅ GroqWhisperClient initialized")
        
        # Test with a sample audio file (if exists)
        test_file = "test_audio.wav"
        if os.path.exists(test_file):
            print(f"\n📁 Testing with {test_file}...")
            text = client.transcribe_file(test_file)
            print(f"✅ Transcription: {text}")
        else:
            print(f"\n⚠️  No test audio file found ({test_file})")
            print("   Skipping file transcription test")
        
        print("\n✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure groq package is installed: pip install groq")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_orchestrator_integration():
    """Test Groq Whisper integration with orchestrator."""
    print("\n" + "=" * 60)
    print("Testing Orchestrator Integration")
    print("=" * 60)
    
    # Check STT_BACKEND setting
    stt_backend = os.getenv("STT_BACKEND", "groq")
    print(f"STT_BACKEND: {stt_backend}")
    
    if stt_backend.lower() != "groq":
        print("⚠️  STT_BACKEND is not set to 'groq'")
        print("   Set STT_BACKEND=groq in .env to use Groq Whisper")
    
    try:
        import asyncio
        from src.orchestrator import Orchestrator
        
        async def test():
            print("\n🔧 Initializing orchestrator...")
            orch = Orchestrator()
            await orch.initialize()
            print("✅ Orchestrator initialized")
            
            print("\n🎤 Initializing STT...")
            await orch.initialize_stt()
            print("✅ STT initialized successfully")
            
            # Check which client is being used
            if hasattr(orch, '_stt_client'):
                client_type = type(orch._stt_client).__name__
                print(f"✅ Using STT client: {client_type}")
                
                if client_type == "GroqWhisperClient":
                    print("✅ Groq Whisper API is active!")
                elif client_type == "WhisperClient":
                    print("⚠️  Local Whisper is active (not Groq)")
                    print("   Set STT_BACKEND=groq in .env to use Groq")
            
            await orch.shutdown()
            print("\n✅ Orchestrator integration test passed!")
        
        asyncio.run(test())
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🚀 Groq Whisper API Integration Test\n")
    
    # Test 1: Direct client test
    test1_passed = test_groq_whisper_client()
    
    # Test 2: Orchestrator integration
    test2_passed = test_orchestrator_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Groq Client Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Orchestrator Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Groq Whisper is ready to use.")
        print("\nNext steps:")
        print("1. Make sure GROQ_API_KEY is set in .env")
        print("2. Make sure STT_BACKEND=groq in .env")
        print("3. Run: python main.py --voice")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
