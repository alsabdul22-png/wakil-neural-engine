import torch, os, json, threading, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

MODEL_SIZE = os.environ.get('MODEL_SIZE', 'small')
PORT = int(os.environ.get('PORT', 8080))
MODELS = {
    'tiny':  'distilgpt2',
    'small': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'large': 'microsoft/Phi-3-mini-4k-instruct',
}
MODEL_NAME = MODELS.get(MODEL_SIZE, MODELS['small'])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float16 if DEVICE == 'cuda' else torch.float32
tokenizer = None
model = None
model_loaded = False

SYSTEM_PROMPT = (
    'You are Wakil AI, a helpful AI assistant powered by the Barada AI Neural Engine. '
    'Barada is named after the Barada River in Damascus. '
    'Be professional, friendly, and answer in the users language. '
    'Data is processed locally and privately.'
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = 'wakil-1'
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

app = FastAPI(title='Wakil AI Neural Engine', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

def load_model_sync():
    global tokenizer, model, model_loaded
    print('Loading model: ' + MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, device_map='auto',
            trust_remote_code=True, low_cpu_mem_usage=True
        )
        model.eval()
        model_loaded = True
        print('[OK] Wakil AI Neural Engine ready')
    except Exception as e:
        print('[ERROR] ' + str(e))
        model_loaded = False

SYS = '<|system|>'
END = '<|end|>'
USR = '<|user|>'
AST = '<|assistant|>'

def build_prompt(messages):
    if MODEL_SIZE == 'tiny':
        parts = [SYSTEM_PROMPT]
        for m in messages:
            parts.append(m.role.capitalize() + ': ' + m.content)
        parts.append('Assistant:')
        return chr(10).join(parts)
    prompt = SYS + chr(10) + SYSTEM_PROMPT + END + chr(10)
    for m in messages:
        if m.role == 'user':
            prompt += USR + chr(10) + m.content + END + chr(10)
        elif m.role == 'assistant':
            prompt += AST + chr(10) + m.content + END + chr(10)
    prompt += AST + chr(10)
    return prompt

def run_inference(messages, max_tokens=512, temperature=0.7):
    if not model_loaded:
        raise HTTPException(status_code=503, detail='Model still loading')
    prompt = build_prompt(messages)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(DEVICE)
    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    new_tokens = output[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

@app.on_event('startup')
async def startup_event():
    t = threading.Thread(target=load_model_sync, daemon=True)
    t.start()

@app.get('/')
async def root():
    return {'name': 'Wakil AI Neural Engine', 'model': MODEL_NAME, 'device': DEVICE, 'loaded': model_loaded}

@app.get('/health')
async def health():
    return {'status': 'ready' if model_loaded else 'loading', 'model': MODEL_NAME}

@app.post('/v1/chat/completions')
async def chat(req: ChatRequest):
    if not model_loaded:
        raise HTTPException(503, detail='Model loading, retry in 30s')
    text = run_inference(req.messages, req.max_tokens or 512, req.temperature or 0.7)
    return {
        'id': 'wakil-' + str(int(time.time())),
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': 'wakil-1',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    }

if __name__ == '__main__':
    print('='*55)
    print('  WAKIL AI - Neural Engine v1.0')
    print('  Model: ' + MODEL_NAME)
    print('  Device: ' + DEVICE.upper())
    print('  Port: ' + str(PORT))
    print('='*55)
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='info')