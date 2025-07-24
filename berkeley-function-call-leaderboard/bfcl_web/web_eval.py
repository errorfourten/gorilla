import os
import sys
import threading
import time
from functools import wraps

from anthropic import Anthropic
from bfcl_eval.constants.eval_config import DOTENV_PATH
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file


### Let's make it pretty ###
def with_spinner(message="Thinking...", spinner_chars="|/-\\", refresh=0.1):
    """
    Decorator that shows a spinner and elapsed time while the wrapped function runs.
    Spinner is cleared when finished or on Ctrl+C.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            stop_event = threading.Event()
            seconds_elapsed = 0
            last_tick = time.time()

            def spin():
                nonlocal seconds_elapsed, last_tick
                i = 0
                while not stop_event.is_set():
                    now = time.time()
                    if now - last_tick >= 1:
                        seconds_elapsed += 1
                        last_tick = now
                    char = spinner_chars[i % len(spinner_chars)]
                    sys.stdout.write(f"\r{char} {message} ({seconds_elapsed} seconds)")
                    sys.stdout.flush()
                    time.sleep(refresh)
                    i += 1
                # Clear the spinner line
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()

            thread = threading.Thread(target=spin, daemon=True)
            thread.start()

            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                raise  # allow Ctrl+C to propagate
            finally:
                stop_event.set()
                thread.join()

        return wrapper

    return decorator


### End of pretty stuff

COSTS = {
    "o3": {"input": 2.00, "output": 8.00},
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}

WEB_SEARCH = True


@with_spinner()
def get_openai_guess(system: str, puzzle: str, model="o3"):
    """
    Calls the OpenAI API to return a response to the puzzle passed in.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tools = []
    if WEB_SEARCH:
        tools.append({"type": "web_search_preview"})

    response = client.responses.create(
        model=model,
        instructions=system,
        input=puzzle,
        tools=tools,
        store=False,
        reasoning={"summary": "auto"},
    )

    reasoning_content = ""
    for item in response.output:
        if item.type == "reasoning":
            for summary in item.summary:
                reasoning_content += summary.text + "\n"
        reasoning_content += "\n"

    input_cost = (
        response.usage.input_tokens / 1000000 * COSTS[model]["input"]
        if response.usage
        else 0.00
    )
    output_cost = (
        response.usage.output_tokens / 1000000 * COSTS[model]["output"]
        if response.usage
        else 0.00
    )

    print(reasoning_content)
    print("=" * 50)
    print(response.output_text)
    print("=" * 50)
    print("Chat id:", response.id)
    print(
        f"Total cost: ${input_cost + output_cost:.3f} (${input_cost:.3f} + ${output_cost:.3f})"
    )
    return response.output_text


def get_claude_guess(system, puzzle, model="claude-sonnet-4-0"):
    """
    Calls the Claude API to return a response to the puzzle passed in.
    """

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tools = []
    if WEB_SEARCH:
        tools.append({"type": "web_search_20250305", "name": "web_search"})

    response = client.messages.create(
        model=model,
        system=system,
        max_tokens=20000,
        tools=tools,
        thinking={"type": "enabled", "budget_tokens": 16000},
        messages=[{"role": "user", "content": puzzle}],
    )

    output_text = ""
    reasoning_content = ""
    for block in response.content:
        if block.type == "text":
            output_text += block.text + "\n"
        elif block.type == "thinking":
            reasoning_content += block.thinking + "\n"

    input_cost = (
        response.usage.input_tokens / 1000000 * COSTS[model]["input"]
        if response.usage
        else 0.00
    )
    output_cost = (
        response.usage.output_tokens / 1000000 * COSTS[model]["output"]
        if response.usage
        else 0.00
    )

    print(reasoning_content)
    print("=" * 50)
    print(output_text)
    print("=" * 50)
    print("Claude Message id:", response.id)
    print(
        f"Total cost: ${input_cost + output_cost:.3f} (${input_cost:.3f} + ${output_cost:.3f})"
    )
    return output_text


def get_gemini_guess(system, puzzle, model="gemini-2.5-pro"):
    """
    Calls the Gemini API to return a response to the puzzle passed in.
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(include_thoughts=True),
    )

    response = client.models.generate_content(
        model=model, contents=puzzle, config=config
    )

    output_text = response.text
    reasoning_content = ""
    if (
        response.candidates
        and response.candidates[0]
        and response.candidates[0].content
        and response.candidates[0].content.parts
    ):
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                reasoning_content += part.text + "\n"
            else:
                print("Answer:")
                print(part.text)
                print()

    input_cost = (
        (response.usage_metadata.prompt_token_count / 1000000 * COSTS[model]["input"])
        if response.usage_metadata and response.usage_metadata.prompt_token_count
        else 0
    )
    output_cost = (
        (
            (
                (response.usage_metadata.candidates_token_count or 0)
                + (response.usage_metadata.thoughts_token_count or 0)
            )
            / 1000000
            * COSTS[model]["output"]
        )
        if response.usage_metadata and response.usage_metadata.prompt_token_count
        else 0
    )

    print(reasoning_content)
    print("=" * 50)
    print(output_text)
    print("=" * 50)
    print("Gemini Message id:", getattr(response, "id", "N/A"))
    print(
        f"Total cost: ${input_cost + output_cost:.3f} (${input_cost:.3f} + ${output_cost:.3f})"
    )
    return output_text


system_prompt = "You are a great puzzle expert who is trying to solve a puzzle given some clues. Feel free to use the web search tool to ensure the highest accuracy of your answer. However, if you aren't sure if the answer, it is fine to reply with 'I am unsure'. If you are unable to come up with an answer, please reply with 'I do not know'."
puzzle = """I am thinking of a U.S. high school that in the 2023\u201324 academic year had roughly 15 students for every full-time teacher, whose outdoors club received a $10,000 Outdoor Equity Fund grant in 2023, and which had about 15 % of its students classified as at risk of dropping out during the early 2020s. What is this school?"""
# puzzle="I am thinking of a company that reported operating income in the low-single-digit millions of dollars during fiscal 2024, recorded about $11 million of operating income in one quarter of 2024, and incurred an operating loss of over $1 million in a quarter of 2023. What is this company"

# get_openai_guess(system_prompt, puzzle, model='o3')
# get_claude_guess(system_prompt, puzzle)
get_gemini_guess(system_prompt, puzzle)
