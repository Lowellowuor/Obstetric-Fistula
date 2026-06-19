import json
import random
from typing import Dict, Optional, List, Tuple
from src.utils.helpers import load_config

class DialogueState:
    def __init__(self, session_id: str, intent: str = None, turn: int = 0):
        self.session_id = session_id
        self.intent = intent
        self.turn = turn
        self.stage = "opening" 
        self.follow_up_type = None
        self.phq9_step = None
        self.phq9_score = 0
        self.phq9_complete = False
        self.history = []  
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "turn": self.turn,
            "stage": self.stage,
            "follow_up_type": self.follow_up_type,
            "phq9_step": self.phq9_step,
            "phq9_score": self.phq9_score,
            "phq9_complete": self.phq9_complete,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data):
        state = cls(data["session_id"])
        state.intent = data.get("intent")
        state.turn = data.get("turn", 0)
        state.stage = data.get("stage", "opening")
        state.follow_up_type = data.get("follow_up_type")
        state.phq9_step = data.get("phq9_step")
        state.phq9_score = data.get("phq9_score", 0)
        state.phq9_complete = data.get("phq9_complete", False)
        state.history = data.get("history", [])
        return state

class DialogueManager:
    def __init__(self, response_file: str = "data/responses_en.json", 
                 phq9: Optional['PHQ9Screener'] = None,
                 config_path: str = "config/phase2_config.yaml"):
        with open(response_file, 'r') as f:
            self.responses = json.load(f)
        self.phq9 = phq9
        self.config = load_config(config_path)
        
        self.confidence_threshold = self.config.get("dialogue", {}).get("confidence_threshold", 0.5)
        self.sessions: Dict[str, DialogueState] = {}
    
    def get_or_create_state(self, session_id: str) -> DialogueState:
        if session_id not in self.sessions:
            self.sessions[session_id] = DialogueState(session_id)
        return self.sessions[session_id]
    
    def process_intent(self, session_id: str, intent: str, confidence: float, 
                       message: str = None) -> Tuple[str, bool]:
        """
        Process the detected intent and return a reply and whether it's crisis.
        Returns: (reply, is_crisis)
        """
        state = self.get_or_create_state(session_id)
        state.turn += 1
        
        
        if intent == "crisis":
            reply = self._get_response("crisis", "opening")
            state.stage = "closed"
            return reply, True
        
        # If we have low confidence, ask clarifying question
        if confidence < self.confidence_threshold:
            reply = "I'm not sure I understood. Could you please rephrase that? I'm here to help."
            state.intent = None  # reset intent, wait for rephrasing
            return reply, False
        
        # New intent or changed intent
        if intent != state.intent:
            state.intent = intent
            state.stage = "opening"
            state.follow_up_type = None
        
        # Build reply based on stage
        if state.stage == "opening":
            reply = self._get_response(intent, "opening")
            state.stage = "follow_up"  # move to follow-up next turn
        elif state.stage == "follow_up":
            if message:
                f_type = self._detect_follow_up_type(intent, message)
                if f_type:
                    state.follow_up_type = f_type
            if state.follow_up_type:
                reply = self._get_response(intent, "follow_up", subtype=state.follow_up_type)
                state.stage = "closing"
            else:
                reply = self._get_response(intent, "closing")
                state.stage = "closed"
        elif state.stage == "closing" or state.stage == "closed":
            reply = self._get_response(intent, "closing")
            state.stage = "closed"
        else:
            reply = self._get_response(intent, "closing")
            state.stage = "closed"
        
        # Update history
        if message:
            state.history.append(("user", message))
        state.history.append(("bot", reply))
        
        return reply, False
    
    def _get_response(self, intent: str, stage: str, subtype: str = None) -> str:
        """Get a random response from the JSON."""
        if intent not in self.responses:
            intent = "greeting"
        stage_data = self.responses[intent].get(stage)
        if not stage_data:
            stage_data = self.responses.get("greeting", {}).get("opening", ["I'm here to help."])
            if isinstance(stage_data, list):
                return random.choice(stage_data)
            return "I'm here to help."
        if isinstance(stage_data, dict):
            if subtype and subtype in stage_data:
                return random.choice(stage_data[subtype])
            else:
                all_subtypes = list(stage_data.keys())
                if all_subtypes:
                    return random.choice(stage_data[all_subtypes[0]])
                else:
                    return "I'm here to help."
        elif isinstance(stage_data, list):
            return random.choice(stage_data)
        else:
            return "I'm here to help."
    
    def _detect_follow_up_type(self, intent: str, message: str) -> Optional[str]:
        """Simple keyword-based detection for follow_up_type."""
        if intent == "coping_request":
            if any(word in message.lower() for word in ["sad", "depressed", "hopeless"]):
                return "sadness"
            if any(word in message.lower() for word in ["anxious", "worried", "scared"]):
                return "anxiety"
            if any(word in message.lower() for word in ["hopeless", "giving up"]):
                return "hopelessness"
            return "sadness"
        elif intent == "stigma_disclosure":
            if "family" in message.lower() or "husband" in message.lower():
                return "family"
            if "community" in message.lower() or "people" in message.lower():
                return "community"
            return "family"
        elif intent == "info_fistula":
            if "recovery" in message.lower() or "heal" in message.lower():
                return "recovery"
            if "surgery" in message.lower():
                return "surgery"
            if "continent" in message.lower() or "leak" in message.lower():
                return "continence"
            return "recovery"
        elif intent == "peer_connection":
            if "location" in message.lower() or "town" in message.lower():
                return "location"
            if "online" in message.lower() or "facebook" in message.lower():
                return "online"
            return "location"
        return None
    
    def reset_state(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def handle_phq9(self, session_id: str, message: str) -> Tuple[Optional[str], bool]:
        """
        Handle PHQ‑9 requests and answers.
        Returns: (reply, handled) – if handled, the reply is the bot's message.
        """
        state = self.get_or_create_state(session_id)
        if not self.phq9:
            return "I'm sorry, the depression screening is not available right now.", False
        
        # Starting PHQ-9
        if "depression test" in message.lower() or "phq" in message.lower():
            first_q = self.phq9.start_session(session_id)
            state.stage = "phq9"
            state.phq9_step = 0
            return first_q, True
        
        # Continuing PHQ-9
        if state.stage == "phq9":
            result = self.phq9.submit_answer(session_id, message)
            if "error" in result:
                return result["error"], True
            if result.get("complete"):
                state.stage = "closed"
                severity = result["severity"]
                reply = f"Thank you. Your PHQ‑9 score is {result['total_score']}, which indicates {severity} depression. "
                if severity in ["moderately_severe", "severe"]:
                    reply += "I strongly recommend you speak with a counsellor. Would you like me to help you find one?"
                else:
                    reply += "Continue to monitor your feelings and reach out for support if needed."
                return reply, True
            else:
                return result["next_question"], True
        
        return None, False  # Not handling PHQ-9