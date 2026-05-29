# Contributing to RAG ChatBot

First off, thank you for considering contributing to RAG ChatBot! It's people like you that make open source such a fantastic community to learn, inspire, and create.

We welcome all contributions, including bug reports, feature requests, documentation improvements, and code changes.

## How Can I Contribute?

### 1. Reporting Bugs
This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report, reproduce the behavior, and find related reports.
*   **Check existing issues:** Before creating a bug report, please check if the issue has already been reported.
*   **Provide clear steps:** Explain exactly how to reproduce the bug.
*   **Include context:** Mention your OS, Python version, and any relevant logs or error messages.

### 2. Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. When creating an enhancement issue, please provide the following:
*   A clear and descriptive title.
*   A detailed description of the proposed feature.
*   The reasoning behind the suggestion and how it would benefit the project.

### 3. Submitting Pull Requests
Please follow these steps when submitting a pull request (PR):
1.  **Fork the repository** and create your branch from `main`.
2.  **Set up your environment:** Install the required dependencies using `pip install -r requirements.txt`.
3.  **Make your changes:** Ensure your code follows the existing style and conventions of the project.
4.  **Test your changes:** Verify that your changes do not break any existing functionality.
5.  **Write clear commit messages:** Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification if possible.
6.  **Submit the PR:** Push your branch to your fork and submit a Pull Request to the `main` branch of this repository.

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RAG-Chatbot.git
   cd RAG-Chatbot
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Code Style
- Try to follow [PEP 8](https://peps.python.org/pep-0008/) guidelines for Python code.
- Keep components modular and well-documented.

Thank you for contributing!
