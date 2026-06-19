import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.dialogue_manager import DialogueManager

def test_dialogue_flow():
    dm = DialogueManager("data/responses_en.json")
    
    reply, crisis = dm.process_intent("s1", "greeting", 0.95)
    assert not crisis
    assert any(word in reply.lower() for word in ["hello", "hi", "welcome"])
    
    reply, crisis = dm.process_intent("s1", "coping_request", 0.9, "I feel sad")
    assert not crisis
    assert "sad" in reply.lower() or "breathe" in reply.lower()
    
    reply, crisis = dm.process_intent("s2", "stigma_disclosure", 0.85, "My family rejects me")
    assert not crisis
    assert "family" in reply.lower() or "understand" in reply.lower()
    
    reply, crisis = dm.process_intent("s3", "crisis", 0.99)
    assert crisis
    assert "helpline" in reply.lower() or "call" in reply.lower()
    
    reply, crisis = dm.process_intent("s4", "greeting", 0.4)
    assert not crisis
    assert "rephrase" in reply.lower()

def test_phq9_integration():
    pass
