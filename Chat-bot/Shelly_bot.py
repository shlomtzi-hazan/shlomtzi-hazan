import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class ShellyBot:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=API_KEY, model=MODEL)
        self.stages = [
            {"question": "What negative, limiting, diminishing, or blocking thought arises when you think about your goal?"},
            {"question": "What negative emotion overwhelms you when you think this thought?"},
            {"question": "Where in your body do you feel this emotion? What physical sensations do you feel?"},
            {"question": "Is this negative thought 100% true?"},
            {"question": "If it’s not true, how does thinking it serve you? What do you gain from it?"},
            {"question": "What positive, empowering, progressive thought comes up when you think about the goal?"},
            {"question": "What positive emotion overwhelms you when you think this new empowering thought?"},
            {"question": "Where do you feel this positive emotion in your body?"}
        ]
        self.stage_idx = 0
        self.history = []
        self.user_name = "friend"  # default fallback name

    def update_stage(self, idx, user_input):
        """
        Validate user input: must be at least 3 words.
        """
        if 0 <= idx < len(self.stages):
            if not user_input.strip() or len(user_input.strip().split()) < 3:
                print("Could you please elaborate a bit more?")
                return False
            return True
        return False

    def chat(self):
        print("Hello, I'm Dr. Shelly Rave, your guide through the 8 Stages of the Bina Method for personal transformation.")
        print("Before we begin, may I know your name?")
        name_input = input("You: ").strip()
        if name_input:
            self.user_name = name_input.split()[0].capitalize()

        print(f"\nNice to meet you, {self.user_name}.")
        print("Let's begin. Please answer each question in turn. Type 'exit' to quit at any time.\n")

        while self.stage_idx < len(self.stages):
            current_q = self.stages[self.stage_idx]["question"]
            print(f"{self.user_name}, {current_q}")

            while True:
                user_input = input("You: ")
                if user_input.strip().lower() == "exit":
                    print(f"Thank you for sharing, {self.user_name}. Take care!")
                    return

                if self.update_stage(self.stage_idx, user_input):
                    self.history.append({
                        "stage": self.stage_idx + 1,
                        "question": current_q,
                        "answer": user_input
                    })
                    self.stage_idx += 1
                    break
                else:
                    print("Please try to reflect a bit more. It's okay to take your time.")

        print(f"\nThank you for completing the Bina Method process with me, {self.user_name}.")
        print("Reflect on your answers and notice any changes in your thoughts, emotions, or sensations.")

if __name__ == "__main__":
    bot = ShellyBot()
    bot.chat()
