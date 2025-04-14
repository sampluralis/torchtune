import gradio as gr
import subprocess
import re

# Adjust these paths to your actual checkpoints
POST_TRAINED_CHECKPOINT = "/tmp/checkpoints/swarm_400m_v4_grpo/epoch_0" # "/home/ubuntu/torchtune/checkpoints"
BASE_CHECKPOINT         = "/home/ubuntu/torchtune/checkpoints"
CONFIG_PATH            = "./swarm-generation.yaml"

def run_cli_inference(prompt: str, checkpoint_dir: str) -> str:
    """
    Calls the CLI command:
      tune run generate --config <CONFIG_PATH> prompt.user="<prompt>"
                        checkpointer.checkpoint_dir=<checkpoint_dir>
    Returns the *entire* stdout as a string.
    """
    cmd = [
        "tune", "run", "generate",
        "--config", CONFIG_PATH,
        f'prompt.user="{prompt}"',
        f'checkpointer.checkpoint_dir={checkpoint_dir}'
    ]
    # Capture both stdout and stderr in a single string
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    # If there's an error with the command or tune, you might want to handle it
    if result.returncode != 0:
        return f"[Error]\n{result.stderr or result.stdout}"
    return result.stderr

def extract_answer(cli_output: str, user_prompt: str) -> str:
    """
    Given the full CLI output and the exact user prompt string,
    return everything AFTER the line that reprints the prompt:
    
        prompt.user="<user_prompt>"
    
    If that line can't be found or there is nothing after it,
    return a fallback string.
    """
    # We'll look for a line containing the prompt exactly like:
    # prompt.user="Hello world"
    prompt_marker = f'Model is initialized with precision torch.bfloat16.'

    lines = cli_output.splitlines()
    found_prompt_line = False
    collected = []

    for line in lines:
        # Once we've found the prompt line, collect everything after it
        if found_prompt_line:
            collected.append(line)
        # Check if this line reprints the prompt
        if prompt_marker in line:
            found_prompt_line = True

    if not collected:
        return cli_output#"[No final output found after prompt]"
    
    # Join all lines after the prompt line
    answer_text = "\n".join(collected).strip()
    return answer_text

def generate_from_two_models(prompt: str):
    """
    1) Runs the post-trained checkpoint with the user's prompt
    2) Runs the base checkpoint with the user's prompt
    3) Returns the answer from each
    """
    # 1. Post-trained model inference
    output_post = run_cli_inference(prompt, POST_TRAINED_CHECKPOINT)
    answer_post = extract_answer(output_post, prompt)

    # 2. Base model inference
    output_base = run_cli_inference(prompt, BASE_CHECKPOINT)
    answer_base = extract_answer(output_base, prompt)

    return answer_post, "Not available at the moment"

# -------------- Gradio Interface --------------

with gr.Blocks() as demo:
    gr.Markdown("# Pluralis-swarm-400mv1")

    with gr.Row():
        prompt_in = gr.Textbox(
            label="Enter your prompt:",
            placeholder="e.g. Give me a good recipe for chicken"
        )

    with gr.Row():
        # Two textboxes to show the answers
        answer_post_out = gr.Textbox(label="Post-Trained Model Output")
        answer_base_out = gr.Textbox(label="Base Model Output")

    generate_button = gr.Button("Generate")

    # On button click, run the function
    generate_button.click(
        fn=generate_from_two_models,
        inputs=prompt_in,
        outputs=[answer_post_out, answer_base_out]
    )

demo.launch(share=True)
