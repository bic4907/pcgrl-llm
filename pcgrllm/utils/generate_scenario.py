import itertools
import json
import random

# Update prompts and base prompt to English
base_prompt_english = (
    "The player needs to obtain the Key and escape through the Door. "
    "Monsters that the player may encounter in the middle are {monsters}, "
    "and the number of possible solutions is {solution_count}."
)

monsters = [["BAT"], ["BAT", "SCORPION"], ["BAT", "SCORPION", "SPIDER"]]
solution_counts = [1, 2, 3]  # Possible solution counts

final_result_with_english_prompts = {
    "base_prompt": base_prompt_english,
    "scenarios": {}
}



for idx, (monster_group, solution_count) in enumerate(itertools.product(monsters, solution_counts), start=1):
    # Generate the prompt
    prompt = base_prompt_english.format(
        monsters=", ".join(monster_group),
        solution_count=solution_count
    )

    final_result_with_english_prompts["scenarios"][idx] = {
        "prompt": prompt,
        "important_tiles": monster_group,
        "n_solutions": solution_count
    }

# Convert to JSON format
final_result_with_english_prompts_json = json.dumps(final_result_with_english_prompts, indent=4)

# Save to JSON
with open("scenario_preset.json", "w") as f:
    f.write(final_result_with_english_prompts_json)