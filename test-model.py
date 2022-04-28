import torch
from transformers.models.openai import OpenAIGPTModel, OpenAIGPTTokenizer

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTModel.from_pretrained('openai-gpt')  # type: OpenAIGPTModel

input_ids = tokenizer('Thanks for your inviting', return_tensors="pt").input_ids  # type: torch.Tensor

for _ in range(100):
    outputs = model.forward(
        input_ids=input_ids,
        return_dict=True
    )
    next_token_embedding = outputs.last_hidden_state[:, -1, :]
    token_embeddings = model.tokens_embed.weight
    next_token_id = torch.argmax(torch.mm(next_token_embedding, token_embeddings.t()))
    input_ids = torch.tensor([input_ids.tolist()[0] + [next_token_id.item()]])
    print(tokenizer.convert_ids_to_tokens(next_token_id.item()), end=' ')
