"""tasknet loads heads with plain Auto*.from_pretrained: register a custom-code model's classes up front."""

from transformers import (AutoConfig, AutoModelForMultipleChoice, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoModelForTokenClassification)

def register_remote(model_name):
    """tasknet loads heads with plain Auto*.from_pretrained: register a custom-code model's classes up front."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    auto_map = getattr(cfg, "auto_map", None) or {}
    for auto in (AutoModelForSequenceClassification, AutoModelForMultipleChoice, AutoModelForQuestionAnswering,
                 AutoModelForTokenClassification):
        if auto.__name__ in auto_map:
            AutoConfig.register(cfg.model_type, type(cfg), exist_ok=True)
            auto.register(type(cfg), get_class_from_dynamic_module(auto_map[auto.__name__], model_name), exist_ok=True)
