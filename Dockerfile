FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir transformers==4.47.* accelerate==1.2.* gradio==6.8.* bitsandbytes==0.45.* peft==0.14.*

COPY web.py .

EXPOSE 7860

CMD ["python", "web.py"]
