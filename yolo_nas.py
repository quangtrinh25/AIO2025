from super_gradients.training import models
from super_gradients.common.object_names import Models

def get_yolo_nas_model(model_name: str, pretrained: bool = True, num_classes: int = 80):
    """
    Retrieve a YOLO-NAS model from the SuperGradients model zoo.

    Args:
        model_name (str): The name of the YOLO-NAS model to retrieve.
        pretrained (bool): Whether to load pretrained weights. Default is True.
        num_classes (int): The number of classes for the model. Default is 80.
        """
    if model_name not in [Models.YOLO_NAS_S, Models.YOLO_NAS_M, Models.YOLO_NAS_L]:
        raise ValueError(f"Model name {model_name} is not a valid YOLO-NAS model.")

    model = models.get(model_name, pretrained=pretrained, num_classes=num_classes)
    return model