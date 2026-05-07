import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class VLMVerifier:
    def __init__(self, config):
        print("🚀 Loading Qwen3 VLM...")

        self.config = config

        self.processor = AutoProcessor.from_pretrained(config["model_path"])

        # device_map streams weights directly from disk → GPU via `accelerate`.
        # This avoids loading the full model into CPU RAM first,
        # keeping GPU usage at ~2.5GB instead of ~5GB from the .to(device) approach.
        # Requires: pip install accelerate
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["model_path"],
            dtype=torch.float16 if config["use_fp16"] else torch.float32,
            device_map=config["device"]
        )

        self.model.eval()
        print("✅ VLM loaded")

    def _parse_result(self, raw: str) -> str:
        """
        Robustly extract FIRE / SMOKE / NONE from raw model output.

        Handles:
          - Qwen3 <think>...</think> reasoning blocks
          - Extra punctuation and surrounding text
          - Multi-word answers (checks every word, not just the last)
        """
        # 1. Strip <think>...</think> blocks (Qwen3 reasoning model)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

        # 2. Uppercase and remove punctuation
        cleaned = cleaned.upper()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)

        # 3. Check words in reverse — answer is usually at the end
        words = cleaned.split()
        for word in reversed(words):
            if word in self.config["valid_labels"]:
                return word

        return "NONE"

    def verify(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.config["prompt"]},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)

            # Decode only newly generated tokens (skip the prompt)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False
                )

            new_tokens = output[:, input_len:]
            raw = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )[0]

            print(f"[VLM RAW] {raw!r}")

            result = self._parse_result(raw)

            print(f"[VLM RESULT] {result}")
            return result

        except Exception as e:
            print(f"❌ VLM Error: {e}")
            return "NONE"