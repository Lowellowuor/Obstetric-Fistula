import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.noise_injection import inject_noise
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

config = load_config("config/phase2_config.yaml")

# English templates per intent
TEMPLATES = {
    "coping_request": [
        "I feel so sad, what can I do?",
        "How do I stop feeling alone?",
        "I need help to cope with this situation.",
        "What should I do when I feel hopeless?",
        "Can you give me some advice to feel better?",
        "I don't know how to handle all these emotions.",
        "Is there any way to stop crying all the time?",
        "I feel empty inside, what is wrong with me?",
        "How can I stay strong when everything is falling apart?",
        "I feel like giving up, but I don't want to."
    ],
    "stigma_disclosure": [
        "My family rejects me because of my condition.",
        "People laugh at me, I feel ashamed.",
        "I can't go out because of the smell.",
        "My husband left me because of this.",
        "I feel like a burden to my community.",
        "Everyone judges me, I have no friends left.",
        "I am so embarrassed to even leave my house.",
        "My neighbours mock me, it hurts so much.",
        "I lost my job because of this illness.",
        "My own children avoid me, it breaks my heart."
    ],
    "info_fistula": [
        "Will I ever be continent again?",
        "How long does recovery take after surgery?",
        "What are the risks of another fistula?",
        "Can I have children after this?",
        "Is there any treatment for incontinence?",
        "What causes fistula?",
        "How do I know if my wound is infected?",
        "Should I see a doctor if I feel pain?",
        "Are there any medications to stop the leakage?",
        "What should I do to prevent complications?"
    ],
    "peer_connection": [
        "Can I talk to other women like me?",
        "Are there support groups near me?",
        "I want to meet someone who understands.",
        "How do I connect with other survivors?",
        "Is there a community of fistula survivors?",
        "I feel so isolated, I need to talk to someone.",
        "Do you know any other women who had this surgery?",
        "I want to join a support group, please help me.",
        "Can you put me in touch with other survivors?",
        "I need a friend who knows what I am going through."
    ],
    "crisis": [
        "I want to kill myself.",
        "I don't want to live anymore.",
        "I am going to end my life tonight.",
        "I feel like dying would be easier.",
        "I have no reason to keep going.",
        "Please help me, I want to die.",
        "I am so tired of suffering, I just want to die.",
        "I have been thinking about suicide.",
        "I want to hurt myself.",
        "I wish I had never survived the surgery."
    ],
    "greeting": [
        "Hello, can you hear me?",
        "Hi, I'm new here.",
        "Good morning, I need some help.",
        "Hey, is anyone there?",
        "Hello, I want to ask something.",
        "Hi there, I am feeling lonely.",
        "Greetings, I need support.",
        "Hello, can we talk?",
        "Hi, I'm a fistula survivor.",
        "Hey, I heard you can help me."
    ]
}

INTENT_LIST = list(TEMPLATES.keys())
INTENT_TO_ID = {intent: idx for idx, intent in enumerate(INTENT_LIST)}

def generate_chat_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic chat messages with noise, English only."""
    data = []
    # Class distribution: more coping and info, fewer crisis (with oversampling)
    intent_weights = [0.25, 0.15, 0.20, 0.15, 0.10, 0.15]  # sum to 1
    for _ in tqdm(range(n_samples), desc="Generating chat data"):
        intent = np.random.choice(INTENT_LIST, p=intent_weights)
        text = random.choice(TEMPLATES[intent])
        # Add noise (typos, shorthand, missing spaces)
        text = inject_noise(text, prob=config['chat']['noise_probability'])
        data.append({
            "text": text,
            "intent": intent,
            "label": INTENT_TO_ID[intent]
        })
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_chat_data(config['chat']['n_samples'])
    df.to_csv("data/synthetic/chat_data.csv", index=False)
    logger.info(f"Saved {len(df)} chat messages to data/synthetic/chat_data.csv")
    print(df['intent'].value_counts())