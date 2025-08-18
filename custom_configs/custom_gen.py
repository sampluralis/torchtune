import torch
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.models.llama3_2 import llama3_2_3b
from torchtune.generation import generate
from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.data import Message

model = llama3_2_3b().cuda()

checkpointer = FullModelHFCheckpointer(
checkpoint_dir="/home/ubuntu/models/Llama-3.2-3B/",
checkpoint_files=[
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    ],
    model_type="LLAMA3_2",
    output_dir="/tmp/torchtune/llama3_8b"
    )
checkpoint = checkpointer.load_checkpoint()
model.load_state_dict(checkpoint["model"])

tokenizer = llama3_tokenizer("/home/ubuntu/models/Llama-3.2-3B/original/tokenizer.model")
messages = [
    Message(role="assistant", content="The funniest joke I've heard is "),
]

#print(messages)
#prompt = tokenizer.encode("The largest economy in the world is", True, False)
prompt = tokenizer({"messages": messages}, inference=True)
input = tokenizer.decode(prompt['tokens'], skip_special_tokens=False)
print(input)
output, logits = generate(model, torch.tensor(prompt['tokens'], device='cuda'), max_generated_tokens=100, pad_id=0)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=False))