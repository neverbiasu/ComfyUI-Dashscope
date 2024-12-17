import os
from dashscope import Generation


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
    RETURN_TYPES = ("String",)

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
