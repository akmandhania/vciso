# VCISO Security IRP Assistant

**VCISO Security IRP Assistant** is an AI-powered tool designed to help organizations and individuals build, review, and test their security Incident Response Plans (IRP). Leveraging Large Language Models (LLMs) and a Gradio web interface, it streamlines the process of managing and improving your security response posture.

**Main Features:**
- **IRP Builder:** Guided creation of an Incident Response Plan.
- **Policy Review & Compliance:** Automated review and compliance checking of your security policies.
- **Policy Simulation & Knowledge Test:** Simulate incidents and test your knowledge of the policies.

## Getting Started

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd viso
```

### 2. Set Up a Python Virtual Environment (Recommended)

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r ../requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the `viso/` directory with your OpenAI API key and any other required keys:

```
OPENAI_API_KEY=sk-...
```

Or export it in your shell:

```sh
export OPENAI_API_KEY=sk-...
```

### 5. Run the App

You can run any part of the Viso app using the `run.py` script:

```sh
python run.py --part part1  # For IRP builder
python run.py --part part2  # For policy review & compliance
python run.py --part part3  # For policy simulation & knowledge test
```

This will launch a Gradio web interface in your browser.

### 6. Usage
- Follow the on-screen instructions to interact with the assistant.
- Use commands like `overview`, `review`, `download`, and `progress` in the chat for additional features.

## Updating & Publishing
- Make sure to update this README if you add new features or dependencies.
- Ensure `plan_outline.txt` and other generated files are in `.gitignore`.

## Troubleshooting
- If you see errors about missing API keys, check your `.env` file or environment variables.
- For dependency issues, try re-creating your virtual environment and reinstalling requirements.

## License
MIT or your chosen license.

## Sample Policy Documents

The `IRCurrentDocuments` directory contains sample policy documents (TXT format) for testing the RAG (Retrieval-Augmented Generation) system in part2.py. You can add, remove, or modify these files as needed for your own organization or experiments. 
>>>>>>> f3ea737 (Initial commit of VCISO Security IRP Assistant)
