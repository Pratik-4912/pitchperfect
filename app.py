import streamlit as st
from typing import Optional
import os

st.set_page_config(page_title="AI Product Description Generator", page_icon="🛍️", layout="centered")

st.title("🛍️ AI Product Description Generator")
st.write("Product name + features टाका → SEO-friendly description, title आणि meta description मिळवा.")

col1, col2 = st.columns([3,1])
with col1:
    product_name = st.text_input("Product Name", placeholder="e.g. Wireless Bluetooth Earbuds")
    features = st.text_area("Key Features (comma separated or new lines)", height=120,
                            placeholder="Noise cancellation\n20 hours battery\nWater-resistant\nBluetooth 5.2")
    extra_keywords = st.text_input("Target keywords (comma separated) — optional", placeholder="wireless earbuds, bluetooth earphones")
    tone = st.selectbox("Tone", ["Professional", "Casual", "Persuasive", "Technical", "Luxury"], index=2)
    length = st.selectbox("Description length", ["Short (40-60 words)", "Medium (80-120 words)", "Long (150-220 words)"], index=1)

with col2:
    st.markdown("### Pricing (suggestion)")
    st.write("• Single description: ₹15\n• Bulk (50): ₹600\n• Monthly unlimited: ₹999")
    st.info("Tip: Offer SEO keyword insertion + bullet points as premium add-on.")

generate = st.button("Generate Description")

st.markdown("---")
st.header("Output")

# --- Model utilities: try HF Inference API if token provided, otherwise use local small model ---
HF_API_KEY = os.getenv("HF_API_KEY")  # optional: set this in Streamlit Cloud secrets or local env

def build_prompt(name: str, feats: str, keywords: str, tone: str, length_choice: str) -> str:
    feats_clean = feats.replace("\n", "; ").strip()
    kw = keywords.strip()
    if length_choice.startswith("Short"):
        word_hint = "about 50 words"
    elif length_choice.startswith("Medium"):
        word_hint = "about 100 words"
    else:
        word_hint = "about 180 words"
    prompt = (
        f"You are an expert e-commerce copywriter. Write an SEO-optimized product description for the following product.\n\n"
        f"Product Name: {name}\n"
        f"Key Features: {feats_clean}\n"
        f"Tone: {tone}\n"
        f"Target Keywords: {kw if kw else 'none specified'}\n"
        f"Length: {word_hint}\n\n"
        "Produce:\n"
        "1) A catchy short Title (6-8 words)\n"
        "2) SEO-friendly product description with 2-3 short bullet points highlighting main features\n"
        "3) A short meta description (max 160 characters)\n\n"
        "Make it persuasive and include keywords naturally. Use simple, clear language suitable for product pages."
    )
    return prompt

# If user wants to use HF Inference API (recommended for stability on cloud), use requests:
def generate_with_hf_inference(prompt: str, model: str="google/flan-t5-small", max_tokens: int=200):
    import requests, json
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature":0.2}}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # HF returns a list or dict depending on model; handle common cases:
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])
    elif isinstance(data, list) and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    else:
        # fallback for some models
        return str(data)

# Local transformer fallback (loads a small model)
def generate_with_transformers_local(prompt: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

if generate:
    if not product_name.strip():
        st.warning("कृपया Product Name भरा.")
    else:
        prompt = build_prompt(product_name, features or "", extra_keywords or "", tone, length)
        st.subheader("Prompt (for debugging / preview)")
        with st.expander("Show prompt"):
            st.code(prompt, language="text")

        with st.spinner("Generating description..."):
            try:
                if HF_API_KEY:
                    raw = generate_with_hf_inference(prompt, model="google/flan-t5-small", max_tokens=220)
                else:
                    raw = generate_with_transformers_local(prompt)
                # Basic post-processing: split into parts if user requested structure
                st.markdown("### ✅ Generated Result")
                st.write(raw)

                # Try to extract Title / bullets / meta if model returned them in order
                st.markdown("---")
                st.subheader("Suggested Title")
                # naive extraction: take first line as title if short
                first_line = raw.strip().splitlines()[0]
                if len(first_line.split()) <= 12:
                    st.success(first_line)
                else:
                    # fallback: create title from product name + 1 feature
                    feat_preview = features.splitlines()[0] if features else ""
                    st.success(f"{product_name} — {feat_preview}".strip(" -"))

                st.subheader("Short Meta Description (<=160 chars)")
                # naive meta: take last 160 chars of first paragraph
                paragraphs = [p for p in raw.split("\n\n") if p.strip()]
                meta = (paragraphs[0][:157] + "...") if paragraphs else (raw[:157] + "...")
                if len(meta) > 160:
                    meta = meta[:157] + "..."
                st.info(meta)

                st.markdown("---")
                st.caption("Tip: Review the output and tweak keywords/tone for best SEO. You can sell each generated description or offer bulk packages.")
            except Exception as e:
                st.error("Error while generating. See console for details.")
                st.exception(e)

