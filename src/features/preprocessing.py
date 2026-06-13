import re

def clean_text(text: str) -> str:
    """Basic cleaning: lower, remove extra spaces, keep alphanumeric and basic punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\.\,\?\!]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text