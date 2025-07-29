import argparse
import os
import json
import concurrent.futures

import xai_sdk
from anthropic import Anthropic
from bfcl_eval.constants.eval_config import PROJECT_ROOT, DOTENV_PATH
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from utils import with_spinner
from xai_sdk.chat import system as xai_system
from xai_sdk.chat import user as xai_user
from xai_sdk.search import SearchParameters

load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file


COSTS = {
    "o3": {"input": 2.00, "output": 8.00},
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "grok-4": {"input": 3.00, "output": 15.00},
}

WEB_SEARCH = True
WEB_ROOT = PROJECT_ROOT / "bfcl_web"

with open(WEB_ROOT / 'prompts.json', 'r') as f:
    prompts = json.load(f)
SYSTEM_PROMPT = prompts['solve_puzzle']['developer']


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


@with_spinner()
def get_grok_guess(system, puzzle, model="grok-4"):
    """
    Calls the Grok API to return a response to the puzzle passed in.
    """

    client = xai_sdk.Client(api_key=os.getenv("GROK_API_KEY"))

    # Prepare search configuration if web search is enabled
    search_parameters = None
    if WEB_SEARCH:
        search_parameters = SearchParameters()

    chat = client.chat.create(
        model=model,
        messages=[xai_system(system), xai_user(puzzle)],
        search_parameters=search_parameters,
    )

    response = chat.sample()

    output_text = response.content
    reasoning_content = response.reasoning_content

    input_cost = response.usage.prompt_tokens / 1000000 * COSTS[model]["input"]
    output_cost = response.usage.completion_tokens / 1000000 * COSTS[model]["output"]
    live_search_cost = response.usage.num_sources_used * 0.025

    print(reasoning_content)
    print("=" * 50)
    print(output_text)
    print("=" * 50)
    print("Grok Message id:", response.id)
    print(
        f"Total cost: ${input_cost + output_cost + live_search_cost:.3f} (${input_cost:.3f} + ${output_cost:.3f})"
    )
    return output_text


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


def process_puzzle_outputs(model_to_run=None):
    """
    Reads puzzle_outputs.json and runs each valid puzzle through the specified handler(s).
    Only processes puzzles where final_puzzle is not "Bad puzzle".
    
    Args:
        model_to_run (str, optional): Which model to run ('openai', 'gemini', 'grok', or None for all)
    """
    
    with open(WEB_ROOT / "puzzle_outputs.json", "r") as f:
        puzzle_data = json.load(f)
    
    # Filter valid puzzles
    valid_puzzles = [
        entry for entry in puzzle_data 
        if entry.get("final_puzzle") != "Bad puzzle"
    ]
    
    all_results = []
    
    markdown_file = WEB_ROOT / "puzzle_summary.md"
    write_markdown_header(markdown_file)

    # Process each valid puzzle
    for idx, puzzle_entry in enumerate(valid_puzzles):
        puzzle = puzzle_entry["final_puzzle"]
        answer = puzzle_entry["answer"]
        
        print("\n" + "="*80)
        print(f"Processing puzzle {idx+1}/{len(valid_puzzles)}")
        print(f"Puzzle: {puzzle}")
        print(f"Expected answer: {answer}")
        print("="*80 + "\n")
        
        # Initialize response variables
        openai_response = None
        gemini_response = None
        grok_response = None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            if model_to_run is None or model_to_run.lower() == 'openai':
                futures[executor.submit(get_openai_guess, SYSTEM_PROMPT, puzzle)] = 'openai'
            if model_to_run is None or model_to_run.lower() == 'gemini':
                futures[executor.submit(get_gemini_guess, SYSTEM_PROMPT, puzzle)] = 'gemini'
            if model_to_run is None or model_to_run.lower() == 'grok':
                futures[executor.submit(get_grok_guess, SYSTEM_PROMPT, puzzle)] = 'grok'

            for future in concurrent.futures.as_completed(futures):
                model_name = futures[future]
                try:
                    response = future.result()
                    if model_name == 'openai':
                        openai_response = response
                    elif model_name == 'gemini':
                        gemini_response = response
                    elif model_name == 'grok':
                        grok_response = response
                except Exception as exc:
                    print(f'{model_name} handler generated an exception: {exc}')
        
        puzzle_result = {
            "puzzle": puzzle,
            "intended_answer": answer,
            "guesses": {
                "openai": openai_response.strip() if openai_response else None,
                "gemini": gemini_response.strip() if gemini_response else None,
                "grok": grok_response.strip() if grok_response else None,
            }
        }
        all_results.append(puzzle_result)
        append_markdown_row(markdown_file, puzzle_result)
        
        # Print results comparison if more than one model was run
        if model_to_run is None:
            print("\n" + "="*80)
            print("Results comparison:")
            print(f"Expected answer: {answer}")
            print(f"OpenAI answer: {openai_response.strip() if openai_response else 'N/A'}")
            print(f"Gemini answer: {gemini_response.strip() if gemini_response else 'N/A'}")
            print(f"Grok answer: {grok_response.strip() if grok_response else 'N/A'}")
            print("="*80 + "\n")

    output_file = WEB_ROOT / 'solved_puzzles.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_file}")

    return all_results


def write_markdown_header(markdown_file):
    """
    Writes the header of the markdown summary file if it doesn't exist.
    """
    if not os.path.exists(markdown_file):
        with open(markdown_file, 'w') as f:
            f.write("# Puzzle Evaluation Summary\n\n")
            f.write("| Puzzle | Intended Answer | OpenAI Guess | Gemini Guess | Grok Guess |\n")
            f.write("|---|---|---|---|---|---|\n")

def append_markdown_row(markdown_file, result):
    """
    Appends a row to the markdown summary file.
    """
    # Helper to compare answers
    def answers_match(guess, intended):
        if guess is None or intended is None:
            return False
        # Simple case-insensitive comparison
        return guess.strip().lower() == intended.strip().lower()

    with open(markdown_file, 'a') as f:
        puzzle = result['puzzle'].replace('\n', '<br/>')
        intended = result['intended_answer']
        guesses = result['guesses']

        openai_guess = guesses.get('openai', 'N/A')
        gemini_guess = guesses.get('gemini', 'N/A')
        grok_guess = guesses.get('grok', 'N/A')

        openai_match = "✅" if answers_match(openai_guess, intended) else "❌"
        gemini_match = "✅" if answers_match(gemini_guess, intended) else "❌"
        grok_match = "✅" if answers_match(grok_guess, intended) else "❌"

        f.write(f"| {puzzle} | {intended} | {openai_guess} {openai_match} | {gemini_guess} {gemini_match} | {grok_guess} {grok_match} |\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run puzzle evaluation")
    parser.add_argument('--model', choices=['openai', 'claude', 'gemini', 'grok'], 
                        help='Specific model to run (default: run all models)')
    parser.add_argument('--single', action='store_true', 
                        help='Run a single test puzzle instead of processing puzzle_outputs.json')
    parser.add_argument('--puzzle', type=str, help='The puzzle text to use with --single mode.')
    args = parser.parse_args()
    
    if args.single:
        if not args.puzzle:
            parser.error("--puzzle is required when using --single")
        # Run a single test puzzle
        puzzle = args.puzzle
        
        if args.model == 'openai':
            get_openai_guess(SYSTEM_PROMPT, puzzle, model='o3')
        elif args.model == 'claude':
            get_claude_guess(SYSTEM_PROMPT, puzzle)
        elif args.model == 'gemini':
            get_gemini_guess(SYSTEM_PROMPT, puzzle)
        elif args.model == 'grok':
            get_grok_guess(SYSTEM_PROMPT, puzzle)
        else:
            # Run all models if no specific model is specified
            get_openai_guess(SYSTEM_PROMPT, puzzle)
            get_claude_guess(SYSTEM_PROMPT, puzzle)
            get_gemini_guess(SYSTEM_PROMPT, puzzle)
            get_grok_guess(SYSTEM_PROMPT, puzzle)
    else:
        # Process puzzles from puzzle_outputs.json
        process_puzzle_outputs(model_to_run=args.model)


if __name__ == "__main__":
    main()
