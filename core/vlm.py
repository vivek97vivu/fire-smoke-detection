from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

class VLMVerifier:
    def __init__(self, config):
        print("🚀 Loading Qwen3 VLM...")

        self.config = config

        self.processor = AutoProcessor.from_pretrained(config["model_path"])

        self.model = AutoModelForImageTextToText.from_pretrained(
            config["model_path"],
            dtype=torch.float16 if config["use_fp16"] else torch.float32,
            device_map=config["device"]
        )

        self.model.eval()

    def verify(self, image_path):
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

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=False
                )

            result = self.processor.batch_decode(
                output,
                skip_special_tokens=True
            )[0]

            result = result.upper().strip()
            print(f"[VLM RAW] {result}")

            # 🔥 Clean output
            words = result.replace(".", "").split()

            if len(words) > 0:
                result = words[-1]

            # ✅ Strict match
            if result in self.config["valid_labels"]:
                return result

            return "NONE"

        except Exception as e:
            print("❌ VLM Error:", e)
            return "NONE"