<div align="center">

# fine-tuning llm

**a small open model, qlora-tuned into a course teaching assistant on a 4 GB GPU**

<img alt="Qwen2.5 1.5B" src="https://img.shields.io/badge/Qwen2.5%201.5B-b6c4ff?style=flat-square&labelColor=1e1f25&logo=huggingface&logoColor=e3e1e9"> <img alt="QLoRA" src="https://img.shields.io/badge/QLoRA-dce1ff?style=flat-square&labelColor=1e1f25"> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-c6c6d0?style=flat-square&labelColor=1e1f25&logo=pytorch&logoColor=e3e1e9"> <img alt="GTX 1650" src="https://img.shields.io/badge/GTX%201650%20·%204%20GB-b6c4ff?style=flat-square&labelColor=1e1f25&logo=nvidia&logoColor=e3e1e9">

| | base | fine-tuned |
|---|---:|---:|
| ROUGE-L | 0.127 | **0.244** |
| trainable parameters | | 2.1M of 1.5B (0.14%) |

</div>

## What's in it

- `generate_data.py`: turns course notes into natural student questions and answers, using Qwen itself
- `audit_data.py`: checks the dataset for repeats and leaks before training
- `train.py`: QLoRA on 4-bit NF4 with double quantisation, adapters on the attention projections
- `evaluate.py`: base vs adapter on concept recall and ROUGE-L
- `app.py`: a Gradio chat to try it

## How it got here

**v1** trained straight on the notes with templated questions like "Explain section 3.3". The model memorised headings and copied note formatting instead of teaching.

**v2** generates the kind of questions students actually ask ("Why is A* guaranteed to find an optimal path?") and answers them conversationally. 663 train, 142 validation and 119 test examples; validation loss fell every epoch (1.086, 1.050, 1.046).

## Run

```bash
git clone https://github.com/ahadarshx/fine-tuning-llm.git
cd fine-tuning-llm/pipeline
pip install -r requirements.txt

python generate_data.py              # notes in data/raw → data/processed
python train.py                      # adapter in outputs/qwen-course-lora-v2
python evaluate.py --model adapter   # or --model base
python app.py                        # http://127.0.0.1:7860
```

## Settings

| | |
|---|---|
| LoRA | `r=8`, `alpha=16`, `dropout=0.05` on `q k v o` projections |
| batch | 1, with 16 gradient accumulation steps |
| sequence | 512 tokens, 3 epochs, learning rate `5e-5` |

## Limits

It still answers some off-topic questions: the dataset had far more answers than refusals. The fix is retrieval, which became [rag-course-assistant](https://github.com/ahadarshx/rag-course-assistant).

<sub>Qwen2.5 · Transformers · PEFT · bitsandbytes · Gradio</sub>
