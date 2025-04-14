from transformers import AutoModelForCausalLM, AutoTokenizer

#TODO: update it to your chosen epoch
trained_model_path = "/tmp/checkpoints/grpo_llama3b/epoch_0"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=trained_model_path,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(trained_model_path, safetensors=True)


# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "Hoe many planets are in the solar system? "
print("Base model output:", generate_text(model, tokenizer, prompt))