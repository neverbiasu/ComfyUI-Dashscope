from .node import DashscopeLLMLoader, DashscopeVLMLoader


NODE_CLASS_MAPPINGS = {
    "DashscopeLLMLoader": DashscopeLLMLoader,
    "DashscopeVLMLoader": DashscopeVLMLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashscopeLLMLoader": "Dashscope LLM Loader",
    "DashscopeVLMLoader": "Dashscope VLM Loader",
}
