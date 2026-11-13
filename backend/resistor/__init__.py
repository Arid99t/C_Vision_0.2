"""Resistor colour-band reader: synthetic training data, model, training, export.

    python -m backend.resistor.render --preview out.png    look at the training data
    python -m backend.resistor.train                       train  -> weights/resistor.pt
    python -m backend.resistor.export                      export -> static/models/resistor.onnx

The exported model is used on the device by backend/static/js/assist/resistor.js.
"""
