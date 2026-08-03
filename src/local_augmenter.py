import random
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import Config


class LocalAugmenter:
    """
    Offline, reproducible data augmentation pipeline.
    Combines Back-Translation + T5 Paraphrase + Semantic Quality Filter.
    No LLM API dependency.
    """

    def __init__(self, device: str = "cpu", cache_dir: Optional[str] = None, lazy_load: bool = True):
        self.device = self._resolve_device(device)
        self.cache_dir = cache_dir
        self.lazy_load = lazy_load

        self.seed = getattr(Config, "AUGMENTER_SEED", 42)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.bt_languages = list(getattr(Config, "BT_LANGUAGES", ["de", "fr", "ru"]))
        self.paraphrase_count = int(getattr(Config, "PARAPHRASE_COUNT", 5))

        self._marian_pairs: Dict[str, Tuple[Tuple[AutoTokenizer, AutoModelForSeq2SeqLM], Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]]] = {}
        self._failed_bt_langs = set()
        self._t5_tokenizer: Optional[AutoTokenizer] = None
        self._t5_model: Optional[AutoModelForSeq2SeqLM] = None
        self._sim_model: Optional[SentenceTransformer] = None

        self._bt_model_names = {
            "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
            "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
            "ru": ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"),
        }

        self._t5_model_name = "Vamsi/T5_Paraphrase_Paws"
        self._sim_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        print(f"LocalAugmenter: initialized on device={self.device}, lazy_load={self.lazy_load}")

        if not self.lazy_load:
            self._preload_models()

    def _resolve_device(self, requested: str) -> str:
        requested = (requested or "cpu").lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _preload_models(self):
        for lang in self.bt_languages:
            self._load_marian_pair(lang)
        self._load_t5()
        self._load_similarity_model()

    def _load_single_seq2seq(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _load_marian_pair(self, mid_lang: str):
        if mid_lang in self._marian_pairs:
            return self._marian_pairs[mid_lang]
        if mid_lang in self._failed_bt_langs:
            return None

        names = self._bt_model_names.get(mid_lang)
        if not names:
            self._failed_bt_langs.add(mid_lang)
            print(f"LocalAugmenter: Unsupported back-translation language '{mid_lang}'.")
            return None

        try:
            print(f"LocalAugmenter: Loading Marian models for '{mid_lang}'...")
            en_to_mid = self._load_single_seq2seq(names[0])
            mid_to_en = self._load_single_seq2seq(names[1])
            self._marian_pairs[mid_lang] = (en_to_mid, mid_to_en)
            return self._marian_pairs[mid_lang]
        except Exception as e:
            self._failed_bt_langs.add(mid_lang)
            print(f"LocalAugmenter: Failed loading Marian pair for '{mid_lang}': {e}. Skipping this language.")
            return None

    def _load_t5(self):
        if self._t5_model is not None and self._t5_tokenizer is not None:
            return self._t5_tokenizer, self._t5_model

        try:
            print("LocalAugmenter: Loading T5 paraphrase model...")
            self._t5_tokenizer = AutoTokenizer.from_pretrained(self._t5_model_name, cache_dir=self.cache_dir)
            self._t5_model = AutoModelForSeq2SeqLM.from_pretrained(self._t5_model_name, cache_dir=self.cache_dir)
            self._t5_model.to(self.device)
            self._t5_model.eval()
            return self._t5_tokenizer, self._t5_model
        except Exception as e:
            print(f"LocalAugmenter: Failed loading T5 model: {e}")
            self._t5_tokenizer = None
            self._t5_model = None
            return None, None

    def _load_similarity_model(self):
        if self._sim_model is not None:
            return self._sim_model

        try:
            print("LocalAugmenter: Loading sentence-transformer quality filter model...")
            self._sim_model = SentenceTransformer(self._sim_model_name, cache_folder=self.cache_dir, device=self.device)
            return self._sim_model
        except Exception as e:
            print(f"LocalAugmenter: Failed loading sentence-transformer model: {e}")
            self._sim_model = None
            return None

    def _translate(self, text: str, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM) -> str:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=4,
                do_sample=False,
                max_length=128,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def back_translate(self, text: str, mid_lang: str = "de") -> str:
        """
        Translate text EN -> mid_lang -> EN.
        Returns back-translated string.
        """
        text = (text or "").strip()
        if not text:
            return ""

        pair = self._load_marian_pair(mid_lang)
        if not pair:
            return ""

        try:
            (tok_en_mid, model_en_mid), (tok_mid_en, model_mid_en) = pair
            mid = self._translate(text, tok_en_mid, model_en_mid)
            if not mid:
                return ""
            back = self._translate(mid, tok_mid_en, model_mid_en)
            return back.strip()
        except Exception as e:
            print(f"LocalAugmenter: back_translate failed for lang='{mid_lang}': {e}")
            return ""

    def paraphrase_t5(self, text: str, num_return: int = 5) -> List[str]:
        """
        Generate paraphrases using T5.
        Returns list of paraphrased strings.
        """
        text = (text or "").strip()
        if not text:
            return []

        tokenizer, model = self._load_t5()
        if tokenizer is None or model is None:
            return []

        prompt = f"paraphrase: {text} </s>"
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=max(7, num_return),
                num_return_sequences=num_return,
                temperature=1.0,
                max_length=128,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = []
        seen = set()
        for t in decoded:
            s = (t or "").strip()
            key = s.lower()
            if s and key not in seen:
                cleaned.append(s)
                seen.add(key)
        return cleaned

    def quality_filter(
        self,
        original: str,
        candidates: List[str],
        low_threshold: float = 0.7,
        high_threshold: float = 0.98,
    ) -> List[str]:
        """
        Filter candidates by semantic similarity to original.
        Keep only: low_threshold <= similarity < high_threshold
        """
        original = (original or "").strip()
        if not original:
            return []

        clean_candidates = []
        seen = set()
        for c in candidates or []:
            s = (c or "").strip()
            k = s.lower()
            if s and k not in seen and k != original.lower():
                clean_candidates.append(s)
                seen.add(k)

        if not clean_candidates:
            return []

        model = self._load_similarity_model()
        if model is None:
            # Fallback: if embedding model unavailable, keep lexical non-duplicates only.
            return clean_candidates

        try:
            emb = model.encode([original] + clean_candidates, convert_to_tensor=True, normalize_embeddings=True)
            original_emb = emb[0]
            cand_emb = emb[1:]
            scores = util.cos_sim(original_emb, cand_emb)[0].detach().cpu().tolist()

            kept = []
            for candidate, score in zip(clean_candidates, scores):
                if low_threshold <= score < high_threshold:
                    kept.append(candidate)
            return kept
        except Exception as e:
            print(f"LocalAugmenter: quality_filter failed, returning lexical-filtered candidates. Error: {e}")
            return clean_candidates

    def _dedupe_texts(self, texts: List[str]) -> List[str]:
        out = []
        seen = set()
        for t in texts:
            s = (t or "").strip()
            k = s.lower()
            if s and k not in seen:
                out.append(s)
                seen.add(k)
        return out

    def generate_intent_data_batch(
        self,
        intent_name: str,
        description: str,
        seeds: List[str],
        total_count: int = 50,
    ) -> List[Dict[str, str]]:
        """
        DROP-IN REPLACEMENT for TeacherAgent.generate_intent_data_batch().
        Returns: [{"text": "...", "label": "intent_name"}, ...]
        """
        total_count = max(1, int(total_count))
        low = float(getattr(Config, "QUALITY_FILTER_LOW", 0.7))
        high = float(getattr(Config, "QUALITY_FILTER_HIGH", 0.98))

        clean_seeds = self._dedupe_texts(seeds or [])
        if not clean_seeds:
            fallback_seed = f"{intent_name.replace('_', ' ')} request"
            if description:
                fallback_seed = f"{fallback_seed}: {description}"
            clean_seeds = [fallback_seed]

        print(
            f"LocalAugmenter: Generating data for intent='{intent_name}' "
            f"with {len(clean_seeds)} seeds (target={total_count})."
        )

        pool: List[str] = []
        per_seed_paraphrases: Dict[str, List[str]] = {}

        # First pass: original + BT(DE/FR/RU) + T5 paraphrase
        for seed in clean_seeds:
            seed_candidates: List[str] = []

            # Keep original seed
            seed_candidates.append(seed)

            # Back-translation variants
            for lang in self.bt_languages:
                bt = self.back_translate(seed, mid_lang=lang)
                if bt:
                    seed_candidates.append(bt)

            # T5 paraphrases
            paras = self.paraphrase_t5(seed, num_return=self.paraphrase_count)
            per_seed_paraphrases[seed] = paras
            seed_candidates.extend(paras)

            # Keep original + filtered augmented forms
            filtered = self.quality_filter(seed, seed_candidates, low_threshold=low, high_threshold=high)
            merged = [seed] + filtered
            pool.extend(self._dedupe_texts(merged))

        pool = self._dedupe_texts(pool)

        # Second pass if still short: back-translate paraphrases
        if len(pool) < total_count:
            print("LocalAugmenter: Running second pass (back-translation over paraphrases)...")
            for seed in clean_seeds:
                if len(pool) >= total_count:
                    break

                paras = per_seed_paraphrases.get(seed, [])
                second_pass_candidates: List[str] = []
                for p in paras:
                    for lang in self.bt_languages:
                        bt2 = self.back_translate(p, mid_lang=lang)
                        if bt2:
                            second_pass_candidates.append(bt2)

                filtered_second = self.quality_filter(
                    seed,
                    second_pass_candidates,
                    low_threshold=low,
                    high_threshold=high,
                )
                pool.extend(filtered_second)
                pool = self._dedupe_texts(pool)

        # Final trim
        final_texts = self._dedupe_texts(pool)[:total_count]
        print(f"LocalAugmenter: Generated {len(final_texts)} samples for '{intent_name}'.")

        return [{"text": t, "label": intent_name} for t in final_texts]

    def augment_seeds(self, intents: List[Dict]) -> List[Dict[str, str]]:
        """
        DROP-IN REPLACEMENT for TeacherAgent.augment_seeds().
        Iterates over intents list and combines generated samples.
        """
        intents = intents or []
        all_samples: List[Dict[str, str]] = []
        for intent in intents:
            title = intent.get("title", "unknown")
            description = intent.get("description", "")
            seeds = intent.get("seeds", [])
            batch = self.generate_intent_data_batch(
                intent_name=title,
                description=description,
                seeds=seeds,
                total_count=getattr(Config, "BOOTSTRAP_SAMPLES_PER_INTENT", 50),
            )
            all_samples.extend(batch)

        return all_samples
