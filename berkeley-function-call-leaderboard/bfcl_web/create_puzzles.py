import datetime
import json
import os
import threading
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from enum import Enum
from typing import List, Optional, Tuple, Type

from bfcl_eval.constants.eval_config import DOTENV_PATH
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ParsedResponse, Response
from pydantic import BaseModel, Field
from utils import with_spinner

load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file

MODEL = "o3"
WEB_SEARCH = True
MAX_RETRY = 3
NUM_FACTS = 5
MAX_THREADS = 10

COSTS = {
    "o3": {"input": 2.00, "output": 8.00},
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CreatedFact(BaseModel):
    fact: str
    source_url: str

    def __str__(self):
        return f"Fact: {self.fact} ({self.source_url})"


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    LOW = "Low"


class PuzzleGuess(BaseModel):
    confidence: ConfidenceLevel
    guess: str


class PuzzleRefinement(BaseModel):
    version: int
    puzzle: str
    response_id: str = Field(exclude=True)
    guess: Optional[str] = None
    confidence: Optional[ConfidenceLevel] = None

    def __str__(self):
        return f"Puzzle: {self.puzzle}"


class Puzzle(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    model: str = MODEL
    answer: str
    original_facts: List[CreatedFact] = Field(default_factory=list)
    revised_facts: List[CreatedFact] = Field(default_factory=list)
    final_puzzle: Optional[str] = None
    refinements: List[PuzzleRefinement] = Field(default_factory=list)
    total_cost: float = 0.0


@with_spinner()
def get_response(
    system: str,
    prompt: str,
    user: Optional[object] = None,
    prev_resp_id: Optional[str] = None,
    schema: Optional[Type[BaseModel]] = None,
) -> ParsedResponse:
    """
    Get a response from the OpenAI API.

    Args:
        system: The system message
        prompt: The prompt to send to the API
        user: Optional user input to format into the prompt
        prev_resp_id: Optional previous response ID for follow-up queries
        schema: Optional Pydantic model to parse the response into

    Returns:
        ParsedResponse
    """
    tools = []
    if WEB_SEARCH:
        tools.append({"type": "web_search_preview"})

    kwargs = {
        "model": MODEL,
        "instructions": system,
        "input": prompt.format(user) if user else prompt,
        "tools": tools,
        "reasoning": {"summary": "auto"},
    }

    if prev_resp_id:
        kwargs["previous_response_id"] = prev_resp_id

    if schema:
        kwargs["text_format"] = schema

    response = client.responses.parse(**kwargs)

    return response


def get_reasoning(response: Response) -> str:
    reasoning_content = ""
    for item in response.output:
        if item.type == "reasoning":
            for summary in item.summary:
                reasoning_content += summary.text
        reasoning_content += "\n"

    return reasoning_content


def get_costs(response: Response) -> float:
    input_cost = (
        response.usage.input_tokens / 1000000 * COSTS[MODEL]["input"]
        if response.usage
        else 0.00
    )
    output_cost = (
        response.usage.output_tokens / 1000000 * COSTS[MODEL]["output"]
        if response.usage
        else 0.00
    )

    return input_cost + output_cost


def create_single_fact(answer: str) -> Optional[Tuple[CreatedFact, float]]:
    """Create a single fact about the answer.

    Args:
        answer: The answer to create a fact about

    Returns:
        A tuple of (fact, cost) if successful, None otherwise
    """
    response: ParsedResponse[CreatedFact] = get_response(
        PROMPTS["fact_finder"]["system"],
        PROMPTS["fact_finder"]["user"],
        answer,
        schema=CreatedFact,
    )

    fact = response.output_parsed
    if not fact:
        return None

    cost = get_costs(response)
    return fact, cost


def create_facts_parallel(answer: str, puzzle: Puzzle) -> None:
    """Create multiple facts in parallel.

    Args:
        answer: The answer to create facts about
        puzzle: The puzzle object to add facts to
    """
    facts_needed = max(0, NUM_FACTS - len(puzzle.original_facts))
    if facts_needed <= 0:
        return

    print(f"Generating {facts_needed} facts for {answer} in parallel...")

    # Thread-safe access to puzzle object
    results_lock = threading.Lock()

    # Process a fact result from a thread
    def process_result(future: Future) -> None:
        result = future.result()
        if result:
            fact, cost = result
            with results_lock:
                puzzle.original_facts.append(fact)
                puzzle.total_cost += cost
            print(f"Generated fact: {fact.fact[:50]}...")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(create_single_fact, answer) for _ in range(facts_needed)
        ]

        for future in futures:
            future.add_done_callback(process_result)

        _, not_done = wait(
            futures,
            timeout=600,  # 10-minute timeout
            return_when=ALL_COMPLETED,
        )

        if not_done:
            print(
                f"Warning: {len(not_done)} fact creation tasks did not complete within the timeout"
            )

    print(
        f"Successfully generated {len(puzzle.original_facts)} facts out of {facts_needed} attempted"
    )


def create_fact(answer: str, puzzle: Puzzle) -> None:
    """Create a fact about the answer and add it to the puzzle."""
    response: ParsedResponse[CreatedFact] = get_response(
        PROMPTS["fact_finder"]["system"],
        PROMPTS["fact_finder"]["user"],
        answer,
        schema=CreatedFact,
    )
    fact = response.output_parsed
    if not fact:
        return None

    puzzle.original_facts.append(fact)
    puzzle.total_cost += get_costs(response)

    # TODO: Do I want to include their reasoning process somewhere?


def rephrase_fact(fact: CreatedFact, puzzle: Puzzle) -> CreatedFact:
    """Rephrase a fact and add it to the puzzle."""
    response: ParsedResponse[CreatedFact] = get_response(
        PROMPTS["rephraser"]["system"],
        PROMPTS["rephraser"]["user"],
        fact,
        schema=CreatedFact,
    )
    revised_fact = response.output_parsed
    if not revised_fact:
        revised_fact = fact

    # TODO: Do I want to include their reasoning process somewhere?

    puzzle.revised_facts.append(revised_fact)
    puzzle.total_cost += get_costs(response)
    return revised_fact


def create_puzzle(puzzle: Puzzle) -> None:
    # Format facts for the puzzle creation
    facts_text = "FACTS:\n"
    for i, fact in enumerate(puzzle.revised_facts, 1):
        facts_text += f"{i}. {fact.fact}\n"

    response = get_response(
        PROMPTS["create_puzzle"]["system"], PROMPTS["create_puzzle"]["user"], facts_text
    )
    puzzle.refinements.append(
        PuzzleRefinement(
            version=0, puzzle=response.output_text, response_id=response.id
        )
    )

    puzzle.total_cost += get_costs(response)


def solve_puzzle(puzzle: Puzzle) -> ConfidenceLevel | None:
    response: ParsedResponse[PuzzleGuess] = get_response(
        PROMPTS["solve_puzzle"]["system"],
        PROMPTS["solve_puzzle"]["user"],
        puzzle.refinements[-1],
        schema=PuzzleGuess,
    )
    puzzle_guess = response.output_parsed
    if puzzle_guess:
        recent_puzzle = puzzle.refinements[-1]
        recent_puzzle.confidence = puzzle_guess.confidence
        recent_puzzle.guess = puzzle_guess.guess

        puzzle.total_cost += get_costs(response)
        return puzzle_guess.confidence


def refine_puzzle(puzzle: Puzzle) -> None:
    text = f"{puzzle.refinements[-1].confidence} - {puzzle.refinements[-1].guess}"
    response = get_response(
        PROMPTS["refine_puzzle"]["system"],
        PROMPTS["refine_puzzle"]["user"],
        text,
        prev_resp_id=puzzle.refinements[-1].response_id,
    )
    puzzle.refinements.append(
        PuzzleRefinement(
            version=len(puzzle.refinements),
            puzzle=response.output_text,
            response_id=response.id,
        )
    )

    puzzle.total_cost += get_costs(response)


def create_puzzle_for_answer(answer: str):
    puzzle = Puzzle(answer=answer)

    create_facts_parallel(answer, puzzle)

    for fact in puzzle.original_facts:
        rephrase_fact(fact, puzzle)

    create_puzzle(puzzle)

    confidence = None
    tries = 0

    while tries < MAX_RETRY:
        confidence = solve_puzzle(puzzle)

        if confidence == ConfidenceLevel.LOW:
            break

        refine_puzzle(puzzle)
        tries += 1

    # If we exited the loop due to MAX_RETRY and not LOW confidence,
    # we need one more solve attempt on the last refinement
    if confidence != ConfidenceLevel.LOW and tries == MAX_RETRY:
        confidence = solve_puzzle(puzzle)

    if confidence == ConfidenceLevel.LOW:
        puzzle.final_puzzle = puzzle.refinements[-1].puzzle
    else:
        puzzle.final_puzzle = "Bad puzzle"

    print(f"Final puzzle for {puzzle.answer}: {puzzle.final_puzzle}")

    # Append to output JSON
    output_filename = "puzzle_outputs.json"

    try:
        with open(output_filename, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, start with empty list
        existing_data = []

    # Convert Puzzle instance to dict and append to existing data
    existing_data.append(puzzle.model_dump())

    # Write all data back to the file
    with open(output_filename, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(
        f"Data for {puzzle.answer} appended to {output_filename} (total entries: {len(existing_data)})"
    )


def main():
    # Read answers from file
    with open("answers.txt", "r") as f:
        answers = [line.strip() for line in f if line.strip()]

    print(f"Found {len(answers)} answers in answers.txt.")

    with open("puzzle_outputs.json", "r") as f:
        existing_data = json.load(f)
        existing_answers = {entry.get("answer", "").lower() for entry in existing_data}

    for answer in answers:
        if answer.lower() in existing_answers:
            print(f"\n=== Skipping puzzle for: {answer} (already exists) ===")
            continue

        print(f"\n=== Creating puzzle for: {answer} ===")
        create_puzzle_for_answer(answer)
        # Add to set to prevent duplicates within the same run
        existing_answers.add(answer.lower())

    print("Done!")


if __name__ == "__main__":
    main()
