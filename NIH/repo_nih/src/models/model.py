"""
Model definition shared between server and clients.
"""
import torch
import torch.nn as nn

def create_model():
    """
    Create a DenseNet121 model for multi-label classification.
    
    Returns:
        torch.nn.Module: DenseNet121 with custom classifier for 7 pathologies
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', 
                           pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1024, 7),
        nn.Sigmoid()
    )
    return model