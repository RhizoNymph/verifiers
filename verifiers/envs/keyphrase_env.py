from typing import List, Dict, Any, Tuple
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.utils import preprocess_dataset
from verifiers.rubrics.keyphrase_rubric import KeyphraseRubric

# Simple prompt for keyphrase extraction
KEYPHRASE_PROMPT = """
You are an expert at identifying key concepts in academic texts.
Extract the most relevant keywords or keyphrases that best summarize the content.
Try to minimize overlap of concepts in keyphrases.
Return your answer as a comma-separated list of phrases.
"""

# Few shot examples could be added here if needed
KEYPHRASE_FEW_SHOT = []

class KeyphraseEnv(SimpleEnv):
    def __init__(self,
                 system_prompt: str = KEYPHRASE_PROMPT,    
                 few_shot: List[Dict[str, str]] = KEYPHRASE_FEW_SHOT,
                 fields: List[str | Tuple[str, ...]] = ["keywords"],
                 **kwargs):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = "krapivin"
        self.dataset = preprocess_dataset(
            dataset_name=self.dataset_name,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot
        ) 
        self.eval_dataset = None
        self.rubric = KeyphraseRubric()
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n)) # type: ignore
        return self.eval_dataset 
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs() 