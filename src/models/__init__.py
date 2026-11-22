from .lstm import LSTMClassifier
from .resnet18 import ResNet18
from .kwt import KWT


def build_model(model_cfg: dict, num_classes: int):
    """
    Factory function to instantiate models based on YAML configuration.
    """
    name = model_cfg.get("name", "").lower()
    if name == "lstm":
        return LSTMClassifier(
            input_size=model_cfg.get("input_size", model_cfg.get("input_dim", 80)),
            hidden_size=model_cfg.get("hidden_size", model_cfg.get("hidden_dim", 64)),
            num_layers=model_cfg.get("num_layers", 2),
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 1),
        )
    elif name == "resnet18":
        return ResNet18(
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 1),
            dropout_rate=model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.5)),
        )
    elif name == "kwt":
        return KWT(
            num_classes=num_classes,
            dim=model_cfg.get("dim", model_cfg.get("embed_dim", 64)),
            depth=model_cfg.get("depth", model_cfg.get("num_layers", 12)),
            heads=model_cfg.get("heads", model_cfg.get("num_heads", 1)),
            mlp_dim=model_cfg.get("mlp_dim", 256),
            dropout=model_cfg.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model name: '{name}'. Supported models: 'lstm', 'resnet18', 'kwt'")


__all__ = ["LSTMClassifier", "ResNet18", "KWT", "build_model"]

