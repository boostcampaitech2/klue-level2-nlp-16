from transformers import Trainer
from loss import *



class CustomTrainer(Trainer):
    """Custom Loss를 적용하기 위한 Trainer"""
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name

    def compute_loss(self, model, inputs, return_outputs=False):
                        
        if "labels" in inputs and self.loss_name != 'default':
            custom_loss = create_criterion(self.loss_name)
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss