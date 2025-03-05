
# Oakie Technologies AI Assistant

A specialized AI assistant designed to help developers follow Oakie Technologies' coding guidelines and best practices.

## Overview

This AI assistant leverages the OpenAI API to provide developers with guidance on:
- Writing code that follows company standards
- Troubleshooting technical issues
- Explaining concepts clearly and accurately
- Providing best practice suggestions

## Features

- Custom-tailored responses based on Oakie Technologies' guidelines
- Support for Python and JavaScript coding conventions
- Conversation memory with ability to reset
- Logging of conversations for future reference

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL_NAME=gpt-4-turbo
   TEMPERATURE=0.7
   MAX_TOKENS=2000
   ENABLE_LOGGING=True
   LOG_PATH=logs/chat_logs.txt
   ```
4. Run the assistant:
   ```
   python main.py
   ```

## Configuration

- `model.json` - Contains company guidelines, tech stack, and coding conventions
- `prompt.jinja2` - Template for the system prompt that guides the AI's behavior
- `.env` - Environment variables for API keys and behavior settings

## Usage

When running the application:
- Type your questions or code snippets for review
- Type 'reset' to start a new conversation
- Type 'exit' to quit the application

## Example Interactions

