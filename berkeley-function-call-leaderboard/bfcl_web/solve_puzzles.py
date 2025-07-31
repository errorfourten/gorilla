import argparse
from enum import Enum
import os
import json
import concurrent.futures
from typing import Dict, Optional

from pydantic import BaseModel, ValidationError
import xai_sdk
from anthropic import Anthropic
from bfcl_eval.constants.eval_config import PROJECT_ROOT, DOTENV_PATH
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from utils import ModelStatusTracker
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


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    LOW = "Low"

class PuzzleGuess(BaseModel):
    confidence: ConfidenceLevel
    guess: str
    process: str
    
    
# Create a custom encoder to handle PuzzleGuess objects
class PuzzleGuessEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PuzzleGuess):
            return o.model_dump()
        return super().default(o)


def get_openai_guess(puzzle: str, model="o3") -> PuzzleGuess:
    """
    Calls the OpenAI API to return a response to the puzzle passed in.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tools = []
    if WEB_SEARCH:
        tools.append({"type": "web_search_preview"})

    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=puzzle,
        tools=tools,
        store=False,
        reasoning={"summary": "auto"},
        text_format=PuzzleGuess
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

    if response.output_parsed:
        return response.output_parsed
    return PuzzleGuess(confidence=ConfidenceLevel.LOW, guess="I am unsure", process="Failed to parse model output.")


def get_grok_guess(puzzle: str, model="grok-4") -> PuzzleGuess:
    """
    Calls the Grok API to return a response to the puzzle passed in.
    """

    client = xai_sdk.Client(api_key=os.getenv("XAI_API_KEY"))

    # Prepare search configuration if web search is enabled
    search_parameters = None
    if WEB_SEARCH:
        search_parameters = SearchParameters()

    chat = client.chat.create(
        model=model,
        messages=[xai_system(SYSTEM_PROMPT), xai_user(puzzle)],
        search_parameters=search_parameters,
    )

    response, puzzle_guess = chat.parse(PuzzleGuess)

    output_text = response.content
    reasoning_content = response.reasoning_content

    input_cost = response.usage.prompt_tokens / 1000000 * COSTS[model]["input"]
    output_cost = response.usage.completion_tokens / 1000000 * COSTS[model]["output"]
    live_search_cost = response.usage.num_sources_used * 0.025

    return puzzle_guess

# IGNORE the following function
def _get_claude_guess(puzzle: str, model="claude-sonnet-4-0"):
    """
    Calls the Claude API to return a response to the puzzle passed in.
    """

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tools = []
    if WEB_SEARCH:
        tools.append({"type": "web_search_20250305", "name": "web_search"})

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
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

    return output_text


def get_gemini_guess(puzzle: str, model="gemini-2.5-pro") -> PuzzleGuess:
    """
    Calls the Gemini API to return a response to the puzzle passed in.
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + f"Please return the output in a JSON format following the supplied schema. {PuzzleGuess.model_json_schema()}",
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        # response_mime_type="application/json",    # Right now, Gemini does not allow JSON with tools
        # response_schema=PuzzleGuess
    )

    max_attempts = 3
    attempts = 0
    parsed_puzzle = None

    while attempts < max_attempts:
        attempts += 1
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
        
        if output_text:
            try:
                parsed_puzzle = PuzzleGuess.model_validate_json(output_text)
                break
                
            except ValidationError:
                try:
                    # This assumes that Gemini returns it in this format:
                    # ```json
                    # { ... }
                    # ```
                    bad_text = output_text.replace("\n", "")[7:-3]
                    parsed_puzzle = PuzzleGuess.model_validate_json(bad_text)
                    break
                except ValidationError as e:
                    print(f"Attempt {attempts}/{max_attempts} failed: {str(e)}")
                    if attempts >= max_attempts:
                        print("Max attempts reached. Returning default PuzzleGuess.")
                        break
                    print(f"Retrying... (attempt {attempts+1}/{max_attempts})")
    
    if parsed_puzzle is None:
        parsed_puzzle = PuzzleGuess(confidence=ConfidenceLevel.LOW, guess="Wrong", process="Failed to parse after multiple attempts.")

    return parsed_puzzle


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
        openai_response: Optional[PuzzleGuess] = None
        gemini_response: Optional[PuzzleGuess] = None
        grok_response: Optional[PuzzleGuess] = None

        # Create list of models to run
        models_to_run = []
        if model_to_run is None or model_to_run.lower() == 'openai':
            models_to_run.append('openai')
        if model_to_run is None or model_to_run.lower() == 'gemini':
            models_to_run.append('gemini')
        if model_to_run is None or model_to_run.lower() == 'grok':
            models_to_run.append('grok')
        
        # Initialize the status tracker
        status_tracker = ModelStatusTracker(models_to_run)
        status_tracker.start()

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                if 'openai' in models_to_run:
                    futures[executor.submit(get_openai_guess, puzzle)] = 'openai'
                if 'gemini' in models_to_run:
                    futures[executor.submit(get_gemini_guess, puzzle)] = 'gemini'
                if 'grok' in models_to_run:
                    futures[executor.submit(get_grok_guess, puzzle)] = 'grok'

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
                        # Mark this model as completed
                        status_tracker.mark_completed(model_name)
                    except Exception as exc:
                        print(f'{model_name} handler generated an exception: {exc}')
                        # Mark as completed even if there was an error
                        status_tracker.mark_completed(model_name)
        finally:
            # Make sure to stop the status tracker
            status_tracker.stop()
        
        # Use the PuzzleGuessEncoder class defined at module level
        
        puzzle_result = {
            "puzzle": puzzle,
            "intended_answer": answer,
            "guesses": {
                "openai": openai_response if openai_response else None,
                "gemini": gemini_response if gemini_response else None,
                "grok": grok_response if grok_response else None,
            }
        }

        append_markdown_row(markdown_file, puzzle_result)
        
        output_file = WEB_ROOT / 'solved_puzzles.json'
        
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                json.dump([puzzle_result], f, indent=2, cls=PuzzleGuessEncoder)
        else:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            
            existing_results.append(puzzle_result)
            
            with open(output_file, 'w') as f:
                json.dump(existing_results, f, indent=2, cls=PuzzleGuessEncoder)
        
        print(f"Result for puzzle {idx+1}/{len(valid_puzzles)} appended to {output_file}")
        
        # Print results comparison if more than one model was run
        if model_to_run is None:
            print("\n" + "="*80)
            print("Results comparison:")
            print(f"Expected answer: {answer}")
            print(f"OpenAI answer: {openai_response.guess if openai_response else 'N/A'}")
            print(f"Gemini answer: {gemini_response.guess if gemini_response else 'N/A'}")
            print(f"Grok answer: {grok_response.guess if grok_response else 'N/A'}")
            print("="*80 + "\n")
    
    print(f"All puzzle results have been processed and saved to {WEB_ROOT / 'solved_puzzles.json'}")


def write_markdown_header(markdown_file):
    """
    Writes the header of the markdown summary file if it doesn't exist.
    """
    if not os.path.exists(markdown_file):
        with open(markdown_file, 'w') as f:
            f.write("# Puzzle Evaluation Summary\n\n")
            f.write("| Puzzle | Intended Answer | OpenAI Guess | Gemini Guess | Grok Guess |\n")
            f.write("|---|---|---|---|---|\n")

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
        guesses: Dict[str, Optional[PuzzleGuess]] = result['guesses']

        def format_guess(guess_obj):
            if isinstance(guess_obj, PuzzleGuess):
                return f"**{guess_obj.confidence}**: {guess_obj.guess}".replace('\n', '<br/>')
            return 'N/A'

        openai_guess_obj = guesses.get('openai')
        gemini_guess_obj = guesses.get('gemini')
        grok_guess_obj = guesses.get('grok')

        openai_guess = format_guess(openai_guess_obj)
        gemini_guess = format_guess(gemini_guess_obj)
        grok_guess = format_guess(grok_guess_obj)

        openai_match = "✅" if answers_match(openai_guess_obj.guess if openai_guess_obj else None, intended) else "❌"
        gemini_match = "✅" if answers_match(gemini_guess_obj.guess if gemini_guess_obj else None, intended) else "❌"
        grok_match = "✅" if answers_match(grok_guess_obj.guess if grok_guess_obj else None, intended) else "❌"

        f.write(f"| {puzzle} | {intended} | {openai_guess} {openai_match} | {gemini_guess} {gemini_match} | {grok_guess} {grok_match} |\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run puzzle evaluation")
    parser.add_argument('--model', choices=['openai', 'gemini', 'grok'], 
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
            print(get_openai_guess(puzzle, model='o3'))
        elif args.model == 'gemini':
            print(get_gemini_guess(puzzle))
        elif args.model == 'grok':
            print(get_grok_guess(puzzle))
        else:
            # Run all models if no specific model is specified
            print(get_openai_guess(puzzle))
            print(get_gemini_guess(puzzle))
            print(get_grok_guess(puzzle))
    else:
        # Process puzzles from puzzle_outputs.json
        process_puzzle_outputs(model_to_run=args.model)


if __name__ == "__main__":
    main()
