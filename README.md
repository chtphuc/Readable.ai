# Readable.ai
🤖 An AI deobfuscator that translates minified/obfuscated JavaScript back into human-readable code.

Have you ever viewed the source of a website, only to be met by a wall of meaningless, machine-generated code?

```javascript
function _0x5dcf(_0x1a2b, _0x3c4d) {
  var _0x7e8f = _0x1a2b["data"][0];
  var _0x9b1a = _0x1a2b["key"];
  if(_0x3c4d > _0x7e8f) {
    for(var _0x5f8d = 0; _0x5f8d < _0x3c4d; _0x5f8d++) {
      console.log(_0x9b1a + _0x5f8d);
    }
  }
  return _0x7e8f;
}
```

This is a nightmare for debugging and analysis. **Readable.ai** is the answer.

-----

## 🔮 The Mission: Translate Chaos into Clarity

Our mission is simple: We use the power of AI (LLMs) to **reverse-engineer** this digital nightmare, restoring logic and human-readable meaning to obfuscated code.

We treat this as a "machine translation" problem:

  * **Source Language:** "Minified" or "obfuscated" code.
  * **Target Language:** The clean, original code.

**Our goal is to turn the cryptic block above into its logical equivalent:**

```javascript
function checkThreshold(config, limit) {
  var firstValue = config["data"][0];
  var prefixKey = config["key"];
  if(limit > firstValue) {
    for(var index = 0; index < limit; index++) {
      console.log(prefixKey + index);
    }
  }
  return firstValue;
}
```

-----

## 🛠️ The Arsenal: Our Methodology

We aren't training a model from scratch. We are fine-tuning a powerful, pre-existing model to specialize in this one task.

  * **Base Model:** `EleutherAI/pythia-410m` (A robust 410-million parameter model).
  * **Training Technique:** **LoRA** (Low-Rank Adaptation). This is a Parameter-Efficient Fine-Tuning (PEFT) technique. It allows us to "teach" the massive base model a new skill by training only a tiny fraction (\< 2%) of its parameters.
  * **Fuel (The Dataset):** This model is trained on a **large-scale, custom-built dataset** featuring tens of thousands of real-world, obfuscated-to-clean code pairs.

-----

## 📈 Status & Roadmap

This project is trained in **multiple stages** due to dataset size and compute limitations:

* **Stage 1 (In Progress):** Trained on the **first 500,000 samples** of the dataset.
    * **Result:** `adapter_v1` (Represents initial learning on a substantial data portion)

* **Stage 2 (Upcoming):** Loading `adapter_v1` and continuing training on the **next segment** of the dataset (e.g., samples 500,001 to 1,000,000).
    * **Result:** `adapter_v2`

* **Stage 3 (Upcoming):** Loading `adapter_v2` and training on the **subsequent segment** until the full dataset is processed.
    * **Result:** `adapter_vX` (The final adapter after processing all data)
-----

## 🚀 How to Use

Your fine-tuned model consists of two parts: the **base model** (`pythia-410m`) and the **adapter** (your trained knowledge). To run inference, you must load the base model, then apply your adapter on top of it.

This is the standard 5-step procedure:

### 1\. Install Dependencies

Ensure you have the necessary libraries installed:

```bash
pip install transformers peft accelerate torch
```

### 2\. Load Base Model & Tokenizer

First, load the original `EleutherAI/pythia-410m` model from Hugging Face. This is the foundation your adapter will be applied to.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "EleutherAI/pythia-410m"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
```

### 3\. Load Your Trained Adapter

Using `PeftModel`, load your trained weights (from the `adapter_model.safetensors` file) and "graft" them onto the base model.

```python
# Provide the path to your trained adapter directory
ADAPTER_PATH = "/path/to/your/trained_adapter/"
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
```

### 4\. Merge for Inference Speed

This is a critical optimization step. The `merge_and_unload()` command permanently fuses the adapter's weights into the base model. This creates a single, highly efficient model and significantly speeds up inference.

```python
model = model.merge_and_unload()
print("✅ Adapter merged. Model is ready!")
```

### 5\. Run Inference

The `model` is now ready. You must format your request as a **prompt** that matches the structure the model was trained on, then call `model.generate()`.

```python
# 1. Define the obfuscated code
obfuscated_code = 'function _0x5dcf(_0x1a2b, _0x3c4d){var _0x7e8f=_0x1a2b["data"][0];var _0x9b1a=_0x1a2b["key"];if(_0x3c4d>_0x7e8f){for(var _0x5f8d=0;_0x5f8d<_0x3c4d;_0x5f8d++){console.log(_0x9b1a+_0x5f8d);}}return _0x7e8f;}'

# 2. Format the prompt
prompt = f"### Input:\n{obfuscated_code}\n\n### Output:\n"

# 3. Tokenize and run generation
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)

# 4. Decode the result
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("--- Deobfuscation Result ---")
print(response_text)
```

-----

## ⚡ Performance & Resource Requirements

Understanding the resource needs is critical. Here is the breakdown based on our tests.

  * **Training (LoRA):** The fine-tuning process is highly efficient. By using LoRA, `fp16`, and the `adafactor` optimizer, a training session (like Stage 1 or 2) requires **approximately 7-8 GB of VRAM**. This makes it perfectly suitable for free Google Colab T4 GPUs.

  * **Inference (After Merging):** This is the key benefit. After using `merge_and_unload()` (Step 4), the adapter is fused into the base model.

      * The final merged model (Base + Adapter) requires the **exact same VRAM as the original `pythia-410m` base model** (approx. 1.8 GB in `fp16`).
      * You pay **no extra VRAM cost** for the adapter's new knowledge.
      * Inference speed is also **identical to the original base model**, as it's a single, unified model.

  * **Portability:** This merged model can be saved and deployed as a standard transformer, without needing the `peft` library for inference.

-----

## ❤️ Support the Project

If you find this project useful, want to support server costs, or just want to buy me a coffee, donations are appreciated.

**BSC (Binance Smart Chain) Address:**
`0x1f7fa6d01f02583b48e0343a9e42cbd408ef3bfb`

-----

## 📄 License
This project is licensed under the MIT License.
