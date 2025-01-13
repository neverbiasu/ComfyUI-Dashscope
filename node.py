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


class DashscopeLLMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (
                    ["qwen_max", "qwen_plus", "qwen_turbo", "qwen_long"],
                    {"default": "qwen_max"},
                ),
                "qwen_max": (
                    [
                        "qwen-max",
                        "qwen-max-latest",
                        "qwen-max-2024-09-19",
                        "qwen-max-2024-04-28",
                        "qwen-max-2024-04-03",
                        "qwen-max-2024-01-07",
                    ],
                    {"default": "qwen-max-latest"},
                ),
                "qwen_plus": (
                    [
                        "qwen-plus",
                        "qwen-plus-latest",
                        "qwen-plus-2024-11-27",
                        "qwen-plus-2024-11-25",
                        "qwen-plus-2024-09-19",
                        "qwen-plus-2024-08-06",
                        "qwen-plus-2024-07-23",
                        "qwen-plus-2024-06-24",
                        "qwen-plus-2024-02-06",
                    ],
                    {"default": "qwen-plus-latest"},
                ),
                "qwen_turbo": (
                    [
                        "qwen-turbo",
                        "qwen-turbo-latest",
                        "qwen-turbo-2024-11-01",
                        "qwen-turbo-2024-09-19",
                        "qwen-turbo-2024-06-24",
                        "qwen-turbo-2024-02-06",
                    ],
                    {"default": "qwen-turbo-latest"},
                ),
                "qwen_long": (["qwen_long"], {"default": "qwen_long"}),
            }
        }

    FUNCTION = "select_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def select_model(self, model_type, qwen_max, qwen_plus, qwen_turbo, qwen_long):
        if model_type == "qwen_max":
            model_version = qwen_max
        elif model_type == "qwen_plus":
            model_version = qwen_plus
        elif model_type == "qwen_turbo":
            model_version = qwen_turbo
        elif model_type == "qwen_long":
            model_version = qwen_long
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        return model_version


class DashscopeVLMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (
                    ["qwen-vl-max", "qwen-vl-plus"],
                    {"default": "qwen-vl-max"},
                ),
                "qwen-vl-max": (
                    [
                        "qwen-vl-max",
                        "qwen-vl-max-latest",
                        "qwen-vl-max-2024-11-19",
                        "qwen-vl-max-2024-10-30",
                        "qwen-vl-max-2024-08-09",
                        "qwen-vl-max-2024-02-01",
                    ],
                    {"default": "qwen-vl-max-latest"},
                ),
                "qwen-vl-plus": (
                    [
                        "qwen-vl-plus",
                        "qwen-vl-plus-latest",
                        "qwen-vl-plus-2024-08-09",
                        "qwen-vl-plus-2023-12-01",
                    ],
                    {"default": "qwen-vl-plus-latest"},
                ),
            }
        }

    FUNCTION = "select_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def select_model(self, model_type, qwen_vl_max, qwen_vl_plus):
        if model_type == "qwen-vl-max":
            model_version = qwen_vl_max
        elif model_type == "qwen-vl-plus":
            model_version = qwen_vl_plus
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        return model_version


class DashscopeModelCaller:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": ("STRING", {"default": None}),
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

    def call_model(self, model_version, system_prompt, user_prompt, image):
        if not model_version:
            raise ValueError("Model version cannot be empty")
        if not system_prompt:
            raise ValueError("System prompt cannot be empty")
        if not user_prompt:
            raise ValueError("User prompt cannot be empty")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        if image == None:
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

        print("API Response:", response)
        
        if "output" not in response:
            raise ValueError(f"Unexpected response format. Response: {response}")

        message_content = response["output"]["choices"][0]["message"]["content"]
        return message_content
