import gradio as gr

from model_runtime import get_runtime


def chat(message: str, history: list[dict]) -> str:
    return get_runtime().generate(message, history)


demo = gr.ChatInterface(
    fn=chat,
    title="MIT 6.034 Course Assistant",
    description=(
        "A fine-tuned Qwen2.5 1.5B assistant. Verify important answers against "
        "the original course notes."
    ),
    examples=[
        "How do BFS and DFS differ?",
        "What makes an A* heuristic admissible?",
        "Does alpha-beta pruning change the minimax result?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
