
def over_100_characters(text: str) -> bool:
    return len(text)>100

def contains_word(text: str) -> bool:
    return "apple" in text.lower()

def function_that_needs_headers(text: str, headers: dict) -> bool:
    return headers['magic-key'] != "123"
