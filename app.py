from flask import Flask, request, jsonify
import torch
from typing import List

from inference import LLaMA, Dialog
app = Flask(__name__)


torch.manual_seed(0)

allow_cuda = False
device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

model = LLaMA.build(
    checkpoints_dir='llama-2-7b-chat/',
    tokenizer_path='tokenizer.model',
    load_model=True,
    max_seq_len=1024,
    max_batch_size=8,
    device=device
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json()
        msg = payload["msg"]
        max_seq_len = int(payload.get("max_seq_len") or 1024)
        top_p = int(payload.get("top_p") or 0.9)
        temperature = int(payload.get("temperature") or 0.6)
        system = str(payload.get("system") or "") 
        
        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": "You are a friendly chatbot named iViVi, you are develop by IVIRSE and always response in Vietnamese. " + system},
                {"role": "user", "content": msg}
            ]
        ]
        
        result = model.chat_completion(
            dialogs,
            max_gen_len=max_seq_len,
            temperature=temperature,
            top_p=top_p
        )
        
        
        return jsonify({
            "code": 200,
            "data": result
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": str(e) or "Smt wrong has been occured!"
        })

if __name__ == "__main__":
    app.run("0.0.0.0", 8000)