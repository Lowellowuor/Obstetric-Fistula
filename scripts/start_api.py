import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict import TriagePredictor
from src.chat.crisis_detector import CrisisDetector
from src.chat.dialogue_manager import DialogueManager, DialogueState
from src.chat.phq9 import PHQ9Screener
from src.data.database import DatabaseManager
from src.data.schemas import PredictionRequest, PredictionResponse
import yaml
from typing import Optional

app = FastAPI(
    title="Fistula Rehabilitation Platform",
    description="AI‑enabled post‑surgical triage and psychosocial support for obstetric fistula",
    version="1.0.0"
)

# Load configs
with open("config/phase1_config.yaml", "r") as f:
    config1 = yaml.safe_load(f)

# Database instance (shared)
db = DatabaseManager(config1['database']['path'])

# ============================================================
# Lazy-loading components
# ============================================================
triage_predictor = None
crisis_detector = None
dialogue_manager = None
phq9 = None

def get_triage_predictor():
    global triage_predictor
    if triage_predictor is None:
        triage_predictor = TriagePredictor(
            model_path="models_artifacts/xgboost_model.pkl",
            db_path=config1['database']['path'],
            embedding_model=config1['features']['text_embedding_model']
        )
    return triage_predictor

def get_chat_components():
    global crisis_detector, dialogue_manager, phq9
    if crisis_detector is None:
        crisis_detector = CrisisDetector(
            model_path="models_artifacts/intent_model.pkl",
            db_path=config1['database']['path']
        )
    if phq9 is None:
        phq9 = PHQ9Screener()
    if dialogue_manager is None:
        dialogue_manager = DialogueManager(
            response_file="data/responses_en.json",
            phq9=phq9
        )
    return crisis_detector, dialogue_manager, phq9

# ============================================================
# Request / Response Models for Chat
# ============================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    is_crisis: bool
    phq9_complete: bool = False
    phq9_severity: Optional[str] = None

# ============================================================
# Endpoints
# ============================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Triage a symptom report (Module A)."""
    pred = get_triage_predictor()
    result = pred.predict(request.patient_id, request.message, request.language)
    return PredictionResponse(
        report_id=result['report_id'],
        triage_class=result['triage_class'],
        confidence=result['confidence'],
        action_recommended=result['action_recommended'],
        explanation=None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat for psychosocial support (Module B).
    Detects intent, handles crisis, administers PHQ‑9, and returns appropriate responses.
    """
    # Ensure session exists in DB
    db.create_chat_session(request.session_id)
    
    # Get chat components
    crisis_detector, dialogue_manager, phq9 = get_chat_components()
    
    # Load existing dialogue state from DB if present
    saved_state = db.load_chat_state(request.session_id)
    if saved_state:
        state = DialogueState.from_dict(saved_state)
        dialogue_manager.sessions[request.session_id] = state

    # 1. Check if the user is responding to PHQ‑9 or starting a new one
    phq9_reply, handled = dialogue_manager.handle_phq9(request.session_id, request.message)
    if handled:
        # Save updated state
        if request.session_id in dialogue_manager.sessions:
            db.save_chat_state(request.session_id, dialogue_manager.sessions[request.session_id].to_dict())
        db.log_chat_message(request.session_id, "user", request.message)
        db.log_chat_message(request.session_id, "bot", phq9_reply)
        
        # Check if PHQ‑9 completed
        phq9_complete = "Thank you" in phq9_reply
        phq9_severity = None
        if phq9_complete:
            # Extract severity from reply (crude)
            if "severe" in phq9_reply:
                phq9_severity = "severe"
            elif "moderately_severe" in phq9_reply:
                phq9_severity = "moderately_severe"
            elif "moderate" in phq9_reply:
                phq9_severity = "moderate"
            elif "mild" in phq9_reply:
                phq9_severity = "mild"
            else:
                phq9_severity = "minimal"
            
            # Save PHQ‑9 score to DB
            # (We need to get the score from the PHQ‑9 session; it's already stored in the screener)
            # For simplicity, we assume the screener session is cleared after completion.
            # We can retrieve the last score from the session state if needed.
            # We'll just update the chat session with severity.
            db.update_phq9_score(request.session_id, 0, phq9_severity)  # score not available here, but we store severity
        
        return ChatResponse(
            reply=phq9_reply,
            is_crisis=False,
            phq9_complete=phq9_complete,
            phq9_severity=phq9_severity
        )

    # 2. Not a PHQ‑9 interaction → run intent detection and dialogue management
    # Embed the message and predict intent
    emb = crisis_detector.embedder.embed([request.message])[0].reshape(1, -1)
    proba = crisis_detector.model.predict_proba(emb)[0]
    intent_idx = int(proba.argmax())
    confidence = float(proba[intent_idx])
    intent = crisis_detector.config['chat']['intents'][intent_idx]
    
    # Double‑check crisis via detector (redundant but safe)
    is_crisis, _ = crisis_detector.detect(request.message)
    if is_crisis:
        intent = "crisis"
        confidence = 1.0  # force high confidence for crisis

    # Process intent through dialogue manager
    reply, is_crisis = dialogue_manager.process_intent(
        request.session_id, intent, confidence, request.message
    )

    # Save updated state and log messages
    if request.session_id in dialogue_manager.sessions:
        db.save_chat_state(request.session_id, dialogue_manager.sessions[request.session_id].to_dict())
    db.log_chat_message(request.session_id, "user", request.message, intent, is_crisis, confidence)
    db.log_chat_message(request.session_id, "bot", reply, intent, is_crisis, confidence)

    return ChatResponse(reply=reply, is_crisis=is_crisis, phq9_complete=False)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "modules": ["triage", "chat"]}

@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str, limit: int = 50):
    """Retrieve chat history for a given session."""
    history = db.get_chat_history(session_id, limit)
    return {"session_id": session_id, "messages": history}

@app.get("/chat/stats")
def get_chat_stats():
    """Retrieve aggregate statistics for chat sessions."""
    stats = db.get_chat_session_stats()
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)