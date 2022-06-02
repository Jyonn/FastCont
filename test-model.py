# import torch
# from transformers.models.openai import OpenAIGPTModel, OpenAIGPTTokenizer
#
# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# model = OpenAIGPTModel.from_pretrained('openai-gpt')  # type: OpenAIGPTModel
#
# input_ids = tokenizer('Thanks for your inviting', return_tensors="pt").input_ids  # type: torch.Tensor
#
# for _ in range(100):
#     outputs = model.forward(
#         input_ids=input_ids,
#         return_dict=True
#     )
#     next_token_embedding = outputs.last_hidden_state[:, -1, :]
#     token_embeddings = model.tokens_embed.weight
#     next_token_id = torch.argmax(torch.mm(next_token_embedding, token_embeddings.t()))
#     input_ids = torch.tensor([input_ids.tolist()[0] + [next_token_id.item()]])
#     print(tokenizer.convert_ids_to_tokens(next_token_id.item()), end=' ')
import json
import os.path

from UniTok import UniDep

depot = UniDep(store_dir='data/ListContUni/zhihu-n1000')
cluster_vocabs = json.load(open(os.path.join(depot.store_dir, 'clusters/cluster_vocab.json')))


count = 0
for sample in depot:
    d = dict(
        k_cluster='k_local',
        p_cluster='p_local',
    )
    for cluster_, local_ in d.items():
        # if len(sample[cluster_]) != len(sample[local_]):
        #     count += 1
        for i, v in enumerate(sample[cluster_]):
            if sample[local_][i] >= cluster_vocabs[v]:
                count += 1
                print(sample)
                print(cluster_, i)
                exit(0)
print(count)
