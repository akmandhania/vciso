from typing import List, Dict, Optional
import os

class VisoPart3:
    def __init__(self):
        self.quiz = [
            {
                'question': 'A ransomware attack is detected. What is the FIRST step?',
                'choices': ['A) Notify law enforcement', 'B) Contain the affected systems', 'C) Pay the ransom', 'D) Ignore it'],
                'answer': 'B',
                'explanation': 'Containment is the first step after detection. See [NIST SP 800-61r3](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf).'
            },
            {
                'question': 'Which section of an IRP covers restoring systems to normal operation?',
                'choices': ['A) Detection', 'B) Recovery', 'C) Containment', 'D) Preparation'],
                'answer': 'B',
                'explanation': 'Recovery covers restoring systems. See [Upwind IRP Guide](https://www.upwind.io/glossary/incident-response-plan-templates-examples#toc-section-3).'
            }
        ]
        self.state = {'current': 0, 'score': 0}
        self.last_result = ''

    def initialize(self):
        self.state = {'current': 0, 'score': 0}
        self.last_result = ''

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        if message.strip().lower() == 'download':
            if not self.last_result:
                return 'No quiz result to download. Please finish the quiz first.'
            path = os.path.join(os.getcwd(), 'plan_outline.txt')
            with open(path, 'w') as f:
                f.write(self.last_result)
            return f"Download your quiz result here: {path} (copy this path to your browser or file explorer)"
        idx = self.state['current']
        if idx < len(self.quiz):
            q = self.quiz[idx]
            if idx == 0 or message.lower() == 'start':
                return f"{q['question']}\n" + '\n'.join(q['choices'])
            if message.strip().upper() == q['answer']:
                self.state['score'] += 1
                response = 'Correct!\n'
            else:
                response = f"Incorrect. {q['explanation']}\n"
            self.state['current'] += 1
            if self.state['current'] < len(self.quiz):
                next_q = self.quiz[self.state['current']]
                response += f"Next: {next_q['question']}\n" + '\n'.join(next_q['choices'])
            else:
                response += f"Quiz complete! Your score: {self.state['score']} out of {len(self.quiz)}."
                self.last_result = f"Quiz complete! Your score: {self.state['score']} out of {len(self.quiz)}."
            return response
        else:
            result = f"Quiz complete! Your score: {self.state['score']} out of {len(self.quiz)}. Type 'start' to try again."
            self.last_result = result
            return result
