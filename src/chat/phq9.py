from src.utils.helpers import load_config

class PHQ9Screener:
    def __init__(self, config_path: str = "config/phase2_config.yaml"):
        self.config = load_config(config_path)
        self.questions = self.config['phq9']['questions']
        self.thresholds = self.config['phq9']['severity_thresholds']
        self.sessions = {}  
    
    def start_session(self, session_id: str) -> str:
        self.sessions[session_id] = {"step": 0, "score": 0}
        return self.questions[0]
    
    def submit_answer(self, session_id: str, answer: str) -> dict:
        if session_id not in self.sessions:
            return {"error": "Session not found. Please start a new session."}
        
        try:
            score = int(answer)
            if score < 0 or score > 3:
                raise ValueError
        except ValueError:
            return {"error": "Please answer with a number between 0 and 3."}
        
        session = self.sessions[session_id]
        session["score"] += score
        session["step"] += 1
        
        if session["step"] >= len(self.questions):
            # Finished
            total = session["score"]
            severity = self._get_severity(total)
            del self.sessions[session_id]  # clear session
            return {"complete": True, "total_score": total, "severity": severity}
        else:
            return {"complete": False, "next_question": self.questions[session["step"]]}
    
    def _get_severity(self, score: int) -> str:
        if score >= self.thresholds["severe"]:
            return "severe"
        elif score >= self.thresholds["moderately_severe"]:
            return "moderately_severe"
        elif score >= self.thresholds["moderate"]:
            return "moderate"
        elif score >= self.thresholds["mild"]:
            return "mild"
        else:
            return "minimal"