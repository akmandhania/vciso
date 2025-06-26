from dotenv import load_dotenv
load_dotenv()

import argparse
import gradio as gr

def main():
    parser = argparse.ArgumentParser(description='Run Viso Security IRP Assistant')
    parser.add_argument('--part', type=str, choices=['part1', 'part2', 'part3'], required=True, help='Which part to run (part1, part2, or part3)')
    args = parser.parse_args()

    if args.part == 'part1':
        from part1 import VisoPart1
        app_class = VisoPart1
        title = "Viso - Part 1: Build Your Incident Response Plan"
        description = "Guides you through building a security incident response plan, with web search and education."
        examples = [
            ["What is an incident response plan?"],
            ["Why do I need an incident response plan?"],
            ["How do I start my IRP?"],
            ["How do I start my incident response plan?"]
        ]
    elif args.part == 'part2':
        from part2 import VisoPart2
        app_class = VisoPart2
        title = "Viso - Part 2: Policy Review & Compliance"
        description = "Upload your company policy docs and see what changes are needed for IRP compliance."
        examples = [["Review my uploaded policy for IRP gaps."]]
    elif args.part == 'part3':
        from part3 import VisoPart3
        app_class = VisoPart3
        title = "Viso - Part 3: Policy Simulation & Knowledge Test"
        description = "Test your knowledge of your IRP with scenario-based quizzes."
        examples = [["Start a simulation."]]
    else:
        raise ValueError(f"Unknown part: {args.part}. Choose from: ['part1', 'part2', 'part3']")

    app = app_class()
    app.initialize()

    def respond(message, history):
        return app.process_message(message, history)

    demo = gr.ChatInterface(
        fn=respond,
        title=title,
        type="messages",
        description=description,
        examples=examples,
        theme=gr.themes.Soft()
    )
    demo.launch()

if __name__ == "__main__":
    main()
