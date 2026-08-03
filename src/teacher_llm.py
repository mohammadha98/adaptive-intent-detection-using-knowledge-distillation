import os
import json
from typing import Dict, List, Optional
import openai
from config import Config

class TeacherAgent:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.base_url=Config.BASE_URL
        self.client = None
        self.intents = self._load_intents_from_file()
        if self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key,base_url=self.base_url)
            except Exception as e:
                print(f"TeacherAgent: Failed to initialize OpenAI client: {e}")
        else:
            print("TeacherAgent: No OpenAI API key found. Using Mock mode.")

    def get_fallback_prediction(self, text: str) -> str:
        """
        Takes user query, returns a predicted label.
        Uses stored intents to constrain the teacher's prediction to valid labels.
        """
        if self.intents:
            # Construct a context-aware prompt with specific intents
            intent_list_str = ""
            for i in self.intents:
                title = i.get('title', 'unknown')
                desc = i.get('description', '')
                seeds = i.get('seeds', [])
                # Include a few seeds for context if available, but keep it brief
                seed_preview = ", ".join(seeds[:2]) if seeds else ""
                intent_list_str += f"- Label: '{title}' | Desc: {desc} | Examples: {seed_preview}\n"

            prompt = f"""
            You are an intent classification expert.
            Classify the following user query into exactly ONE of the provided intent labels.
            
            Valid Intents:
            {intent_list_str}
            - Label: 'out_of_scope' | Desc: Query matches none of the above topics.

            Query: "{text}"
            
            Task: Return ONLY the exact label string (e.g., 'balance_check'). Do not add punctuation or explanation.
            """
        else:
            # Fallback if no intents are stored (e.g. before bootstrap)
            prompt = f"""
            Classify the following user query into a short intent label (e.g., 'balance', 'transfer', 'support').
            Query: "{text}"
            Return ONLY the label.
            """
        
        response = self._call_llm(prompt)
        if response:
            return response.strip().lower()
        
        return "mock_intent"

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.client:
            return None
            
        try:
            response = self.client.chat.completions.create(
                model=Config.TEACHER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"TeacherAgent: OpenAI API call failed: {e}")
            return None

    def _call_llm_json(self, prompt: str) -> Optional[List[str]]:
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model=Config.TEACHER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Return ONLY a valid JSON array of strings. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400,
            )
            content = response.choices[0].message.content or ""
            cleaned = content.replace("```", "").replace("json", "").strip()
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]

            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
            except Exception:
                lines = [l.strip().strip('"').strip("'") for l in cleaned.splitlines() if l.strip()]
                if lines:
                    return lines
        except Exception as e:
            print(f"TeacherAgent: JSON LLM call failed: {e}")

        return None

    def generate_intent_data_batch(
        self,
        intent_name: str,
        description: str,
        seeds: List[str],
        total_count: int = 50,
    ) -> List[Dict[str, str]]:
        """LLM-based augmentation (temporary fallback when local HF models are unavailable)."""
        styles = [
            "Formal and polite",
            "Casual/Slang (short)",
            "Urgent/Frustrated",
            "Indirect/Questioning",
            "With typos/grammar errors",
        ]
        samples_per_style = max(1, total_count // len(styles))
        all_data: List[Dict[str, str]] = []

        for style in styles:
            prompt = (
                f"You are a synthetic data generator. "
                f"Generate {samples_per_style} distinct training examples for the intent '{intent_name}' "
                f"(Description: {description}). The examples must be in the style: '{style}'. "
                f"Use these reference seeds as guidance: {seeds[:3]}. "
                f"Return ONLY a JSON list of strings."
            )
            examples = self._call_llm_json(prompt)

            if not examples:
                base = seeds if seeds else [f"{intent_name} request"]
                examples = [f"{base[i % len(base)]} [{style}]" for i in range(samples_per_style)]

            for t in examples[:samples_per_style]:
                txt = str(t).strip()
                if txt:
                    all_data.append({"text": txt, "label": intent_name})

        # Ensure exact size
        if len(all_data) < total_count:
            base = seeds if seeds else [f"{intent_name} request"]
            i = 0
            while len(all_data) < total_count:
                all_data.append({"text": f"{base[i % len(base)]} [extra]", "label": intent_name})
                i += 1

        return all_data[:total_count]

    def augment_seeds(self, intents: List[Dict]) -> List[Dict[str, str]]:
        """Convenience wrapper to augment all intents via LLM-based pipeline."""
        intents = intents or []
        all_data: List[Dict[str, str]] = []

        for intent in intents:
            batch = self.generate_intent_data_batch(
                intent_name=intent.get("title", "unknown"),
                description=intent.get("description", ""),
                seeds=intent.get("seeds", []),
                total_count=Config.BOOTSTRAP_SAMPLES_PER_INTENT,
            )
            all_data.extend(batch)

        return all_data

    def _load_intents_from_file(self) -> List[Dict]:
        """Load intents from entered_intents.json file."""
        intents_path = "data/entered_intents.json"
        if os.path.exists(intents_path):
            try:
                with open(intents_path, "r", encoding="utf-8") as f:
                    loaded_intents = json.load(f)
                    if isinstance(loaded_intents, list):
                        return loaded_intents
            except Exception as e:
                print(f"TeacherAgent: Error loading intents from file: {e}")
        return []
