from flask import Flask, request, jsonify
import torch

from inference import LLaMA
app = Flask(__name__)


torch.manual_seed(0)

allow_cuda = False
device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

model = LLaMA.build(
    checkpoints_dir='llama-2-7b/',
    tokenizer_path='tokenizer.model',
    load_model=True,
    max_seq_len=1024,
    max_batch_size=1,
    device=device
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json()
        msg = payload["msg"]
        
        prompts = [
            msg
        ]
        
        out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
        assert len(out_texts) == len(prompts)
        res = ""
        for i in range(len(out_texts)):
            res += out_texts[i]
        
        return jsonify({
            "code": 200,
            "data": res
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": str(e) or "Smt wrong has been occured!"
        })

if __name__ == "__main__":
    app.run("0.0.0.0", 8000)