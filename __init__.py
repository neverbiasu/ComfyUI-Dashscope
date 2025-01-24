from .node import (
    DashscopeLLMLoader,
    DashscopeVLMLoader,
    DashscopeModelCaller,
    DashscopeOCRCaller,
    DashscopeEmoCaller,
)


NODE_CLASS_MAPPINGS = {
    "DashscopeLLMLoader": DashscopeLLMLoader,
    "DashscopeVLMLoader": DashscopeVLMLoader,
    "DashscopeModelCaller": DashscopeModelCaller,
    "DashscopeOCRCaller": DashscopeOCRCaller,
    "DashscopeEmoCaller": DashscopeEmoCaller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashscopeLLMLoader": "Dashscope LLM Loader",
    "DashscopeVLMLoader": "Dashscope VLM Loader",
    "DashscopeModelCaller": "Dashscope Model Caller",
    "DashscopeOCRCaller": "Dashscope OCR Caller",
    "DashscopeEmoCaller": "Dashscope Emotion Caller",
}
