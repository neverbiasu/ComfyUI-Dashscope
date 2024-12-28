from .node import DashscopeLLMLoader, DashscopeVLMLoader, DashscopeModelCaller


NODE_CLASS_MAPPINGS = {
    "DashscopeLLMLoader": DashscopeLLMLoader,
    "DashscopeVLMLoader": DashscopeVLMLoader,
    "DashscopeModelCaller": DashscopeModelCaller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashscopeLLMLoader": "Dashscope LLM Loader",
    "DashscopeVLMLoader": "Dashscope VLM Loader",
    "DashscopeModelCaller": "Dashscope Model Caller",
}
