import os
import random
import string
from dashscope import Generation, MultiModalConversation
import torchvision.transforms as transforms


def generate_random_image_name():
    random_number = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"image_{random_number}.png"


def get_image_url(image):
    image = image.squeeze(0)
    image = image.permute(2, 0, 1)
    image = transforms.ToPILImage()(image)

    image_name = generate_random_image_name()
    image.save(image_name)
    return image_name


def get_model_versions(model_type: str) -> list[str]:
    model_versions = []
    current_group = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_versions_file = os.path.join(
        current_dir, "model_versions", f"{model_type}.txt"
    )

    with open(model_versions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_group = line[1:-1]
                continue

            model_versions.append(line)

    return model_versions


class DashscopeLLMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_versions = get_model_versions("llm")
        return {
            "required": {
                "model_version": (
                    model_versions,
                    {"default": model_versions[0]},
                ),
            }
        }

    FUNCTION = "select_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def select_model(self, model_version) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")
        return (model_version,)


class DashscopeVLMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_versions = get_model_versions("vlm")
        return {
            "required": {
                "model_version": (
                    model_versions,
                    {"default": model_versions[0]},
                ),
            }
        }

    FUNCTION = "select_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def select_model(self, model_version) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")
        return (model_version,)


class DashscopeModelCaller:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": ("STRING",),
                "system_prompt": (
                    "STRING",
                    {"default": "You are a helpful assistant."},
                ),
                "user_prompt": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
            },
        }

    FUNCTION = "call_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def call_model(
        self, model_version, system_prompt, user_prompt, image
    ) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")
        if not system_prompt:
            raise ValueError("System prompt cannot be empty")
        if not user_prompt:
            raise ValueError("User prompt cannot be empty")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        if image is None:
            print("Call the LLM model")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = Generation.call(
                api_key=api_key,
                model=model_version,
                messages=messages,
                result_format="message",
            )
        else:
            print("Call the VLM model")
            image_url = get_image_url(image)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": user_prompt},
                    ],
                },
            ]
            response = MultiModalConversation.call(
                api_key=api_key,
                model=model_version,
                messages=messages,
                result_format="message",
            )

        if response is None:
            raise ValueError("API call returned None")

        message_content = response.output.choices[0].message.content[0]["text"]
        return (message_content,)
