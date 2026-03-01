FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir transformers accelerate gradio

# Download model into image
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b-deduped'); \
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-2.8b-deduped', use_safetensors=True)"

COPY web.py .

EXPOSE 7860

CMD ["python", "web.py"]
