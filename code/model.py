import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import get_activation
from transformers import ( AutoModelForSequenceClassification ,
                            AutoConfig,
                            ElectraConfig,
                            ElectraPreTrainedModel,
                            ElectraModel,
                            ElectraPreTrainedModel,
                            XLMRobertaForSequenceClassification,
                            XLMRobertaConfig,
                            RobertaModel,
                            )

def load_model(args, Model_dir='./best.py'):
    if args.inference==False:
        if args.model_name == 'klue/bert-base':
            model_confg=AutoConfig.from_pretrained(args.model_name)
            model_confg.num_labels=30
            model=AutoModelForSequenceClassification.from_pretrained(args.model_name,config=model_confg)
            return model
        
        if args.model_name=='monologg/koelectra-base-v3-discriminator':
            koelectra_config = ElectraConfig.from_pretrained(args.model_name)
            model=koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_name,config=koelectra_config,)
            return model

        if args.model_name == 'xlm-roberta-base':
            xlm_roberta_config=XLMRobertaConfig.from_pretrained(args.model_name)
            xlm_roberta_config.num_labels=30
            model=XLMRobertaForSequenceClassification.from_pretrained(args.model_name, config=xlm_roberta_config)
            return model

        if args.model_name == 'klue/roberta-base':
            model_confg=AutoConfig.from_pretrained(args.model_name)
            model_confg.num_labels=30
            model=AutoModelForSequenceClassification.from_pretrained(args.model_name,config=model_confg)
            return model

        if args.model_name == 'klue/roberta-large':
            model_confg=AutoConfig.from_pretrained(args.model_name)
            model_confg.num_labels=30
            model=AutoModelForSequenceClassification.from_pretrained(args.model_name,config=model_confg)
            return model

        if args.model_name == 'xlm-roberta-large':
            xlm_roberta_config=XLMRobertaConfig.from_pretrained(args.model_name)
            xlm_roberta_config.num_labels=30
            model=XLMRobertaForSequenceClassification.from_pretrained(args.model_name, config=xlm_roberta_config)
            return model

    else:
        if args.model_name == 'klue/bert-base':
            model=AutoModelForSequenceClassification.from_pretrained(Model_dir)
            return model

        if args.model_name == 'klue/roberta-base':
            model=AutoModelForSequenceClassification.from_pretrained(Model_dir)
            return model

        if args.model_name == 'klue/roberta-large':
            model=AutoModelForSequenceClassification.from_pretrained(Model_dir)
            return model
        
        if args.model_name=='monologg/koelectra-base-v3-discriminator':
            koelectra_config = ElectraConfig.from_pretrained(args.model_name)
            model=koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=Model_dir,config=koelectra_config,)
            return model

        if args.model_name == 'xlm-roberta-base':
            model=AutoModelForSequenceClassification.from_pretrained(Model_dir)
            return model
        


class ElectraClassificationHead(nn.Module):
    def __init__(self, config, num_labels=30):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size,num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :] 
        x = self.out_proj(x)
        return x


class koElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config, num_labels=30):
        super().__init__(config)
        self.num_labels = num_labels
        self.model = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config, num_labels)

        self.init_weights()

    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          labels=None,
          output_attentions=None,
          output_hidden_states=None,
    ):

        discriminator_hidden_states = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )
    
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + discriminator_hidden_states[1:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

