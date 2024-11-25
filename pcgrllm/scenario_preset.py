import json
import os
from typing import List, Dict


class Scenario:
    def __init__(self, prompt: str, important_tiles: List[str]):
        self.prompt = prompt
        self.important_tiles = important_tiles

    def __repr__(self):
        return f"Scenario(prompt={self.prompt!r}, important_tiles={self.important_tiles!r})"


class ScenarioPreset:
    def __init__(self, base_prompt: str = None, scenarios: Dict[str, Scenario] = None):
        if base_prompt is None and scenarios is None:
            # 기본 파일 경로 설정
            current_dir = os.path.dirname(__file__)
            file_path = os.path.join(current_dir, "scenario", "scenario_preset.json")

            if os.path.exists(file_path):
                # 파일 로드
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                scenarios = {key: Scenario(**value) for key, value in data.get("scenarios", {}).items()}
                base_prompt = data["base_prompt"]
            else:
                raise FileNotFoundError(f"The file {file_path} does not exist.")

        self.base_prompt = base_prompt
        self.scenarios = scenarios

    def __repr__(self):
        return f"ScenarioPreset(base_prompt={self.base_prompt!r}, scenarios={self.scenarios!r})"


# 사용 예시
if __name__ == "__main__":
    try:
        # 인자 없이 초기화
        preset = ScenarioPreset()
        print(preset)
    except FileNotFoundError as e:
        print(e)
