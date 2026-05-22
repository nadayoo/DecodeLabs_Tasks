# ============================================================
#   Project 1: Rule-Based AI Chatbot
#   DecodeLabs Industrial Training | Batch 2026
#   Author: Nada Wael
# ============================================================

import random
import time


responses = {
    "hello":        ["Hey there! How can I help?",
                     "Hello! Ask me anything.",
                     "Hi! DecodeLabs Bot is ready. "],
    "hi":           ["Hi! What's on your mind?",
                     "Hey! How can I assist?"],
    "how are you":  ["I'm just code, but running at 100%! ",
                     "Fully operational and ready to help!"],
    "what is ai":   ["AI is the simulation of human intelligence by machines.",
                     "Artificial Intelligence: teaching machines to think and decide."],
    "what is ml":   ["Machine Learning is a subset of AI where systems learn from data "
                     "instead of being explicitly programmed."],
    "who made you": ["A DecodeLabs intern built me as Project 1...maybe that's you! "],
    "about":        ["I'm a rule-based chatbot built for DecodeLabs Batch 2026. "
                     "Pure Python, no ML...just logic! "],
    "joke":         ["Why do programmers prefer dark mode? Because light attracts bugs! ",
                     "Why did the AI break up with the algorithm? Too predictable. "],
    "motivate":     ["Keep going! Every expert was once a beginner. ",
                     "One project at a time. You're building something real! "],
    "help":         ["I can talk about: hello, how are you, what is ai, "
                     "what is ml, joke, motivate, about. Type 'exit' to quit."],
    "bye":          ["Goodbye! Keep building. ", "See you next time! "],
    "exit":         ["Shutting down... Goodbye! "],
}

FALLBACK = [
    "I don't know that one yet. Type 'help' to see what I can do.",
    "Hmm, I'm not sure about that. Try typing 'help'.",
]

EXIT_WORDS = {"bye", "exit", "quit"}


def type_effect(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.02)
    print()


def get_response(user_input):
    # Check for keyword match anywhere in the input
    for key in responses:
        if key in user_input:
            return random.choice(responses[key]), key
    return random.choice(FALLBACK), None


print("\n" + "=" * 45)
print("  DecodeLabs Chatbot | Project 1")
print("   Type 'help' for options, 'exit' to quit")
print("=" * 45 + "\n")

while True:
    # PHASE 1: Input & Sanitization
    raw = input("You: ")
    clean = raw.lower().strip()

    if not clean:
        continue

    # PHASE 2: Lookup
    reply, intent = get_response(clean)

    # PHASE 3: Output
    type_effect(f"Bot: {reply}")

    # Exit check
    if intent in EXIT_WORDS:
        break