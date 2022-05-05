import transformers

major_version = transformers.__version__.split('.')[0]

if major_version >= '4':
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as BertOutput
    from transformers.models.bert.modeling_bert import BertModel
else:
    from transformers.modeling_outputs import BaseModelOutputWithPooling as BertOutput
    from transformers.modeling_bert import BertModel
