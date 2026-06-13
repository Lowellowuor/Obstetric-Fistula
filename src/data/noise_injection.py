import random

def inject_noise(text: str, prob: float = 0.4) -> str:
    if random.random() > prob:
        return text.lower()
    noise_type = random.choice(['typo','shorthand','nospace','repeat','caps'])
    text = text.lower()
    if noise_type == 'typo' and len(text) > 3:
        idx = random.randint(0, len(text)-2)
        text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
    elif noise_type == 'shorthand':
        replacements = {' and ':' & ', ' you ':' u ', ' for ':' 4 ', ' are ':' r '}
        for old, new in replacements.items():
            text = text.replace(old, new)
    elif noise_type == 'nospace':
        words = text.split()
        if len(words) > 1:
            idx = random.randint(0, len(words)-2)
            words[idx] += words[idx+1]
            del words[idx+1]
            text = ' '.join(words)
    elif noise_type == 'repeat' and len(text) > 2:
        idx = random.randint(0, len(text)-1)
        text = text[:idx] + text[idx]*2 + text[idx+1:]
    elif noise_type == 'caps':
        # random capitalization
        text = ''.join(c.upper() if random.random() < 0.1 else c for c in text)
    return text