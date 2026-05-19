"""
VLM feature extractor — mean pooling sobre hidden states do LM.

Famílias suportadas:
  internvl  — InternVL3-2B (mesmo pipeline do CaVL-Doc)
  qwen2vl   — Qwen2.5-VL-2B-Instruct
  gemma     — google/gemma-3n-e2b-it         (Gemma 4 ~2B multimodal)
  ministral — mistralai/Ministral-3-3B-Base-2512  (LM 3.4B + vision 0.4B)

Interface pública de cada extractor:
  extract_image(image, prompt, layers) → dict[layer_idx → np.ndarray (dim,)]
  extract_text(text, layers)           → dict[layer_idx → np.ndarray (dim,)]

  layers: tupla de índices de camada — 0 = embedding layer, -1 = última camada.
  A diferença entre layer 0 e layer -1 mede o quanto o LM refina a representação.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilitário de quantização
# ---------------------------------------------------------------------------

def _quant_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _mean_pool_layer(hidden_states: tuple, layer_idx: int, token_mask: torch.Tensor | None) -> torch.Tensor:
    """
    hidden_states: tuple de tensores [1, seq, dim] (output_hidden_states=True)
    layer_idx: 0 = embedding layer, -1 = última camada
    token_mask: bool [1, seq] — True nas posições a pooler (None = todos)
    Retorna tensor [dim] float32 na CPU.
    """
    h = hidden_states[layer_idx]  # [1, seq, dim]
    if token_mask is not None and token_mask.any():
        vt = h[token_mask].mean(dim=0)
    else:
        vt = h[0].mean(dim=0)
    return vt.cpu().float()


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseExtractor:
    def __init__(self, hf_path: str, load_in_4bit: bool = True, device: str = "cuda"):
        self.hf_path = hf_path
        self.load_in_4bit = load_in_4bit
        self.device = device

    def load(self):
        raise NotImplementedError

    def extract_image(
        self,
        image: Image.Image,
        prompt: str,
        layers: tuple[int, ...] = (0, -1),
    ) -> dict[int, np.ndarray]:
        """
        Retorna {layer_idx: embedding float32 (dim,)}.
        Mean pool sobre os tokens visuais no hidden state de cada camada.
        """
        raise NotImplementedError

    def extract_text(
        self,
        text: str,
        layers: tuple[int, ...] = (0, -1),
    ) -> dict[int, np.ndarray]:
        """
        Retorna {layer_idx: embedding float32 (dim,)}.
        Entrada só texto (sem imagem) — usado para embeddings das classes.
        """
        raise NotImplementedError

    def generate_text(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 40,
    ) -> str:
        """
        Gera resposta textual do modelo dado image + prompt.
        Usado no modo de classificação generativa.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InternVL3
# ---------------------------------------------------------------------------

class InternVLExtractor(BaseExtractor):
    INPUT_SIZE = 448
    MAX_NUM = 6  # same as CaVL-Doc; avoids patch/token count mismatch with larger values

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cavl_doc.data.transforms import dynamic_preprocess, build_transform
        from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

        quant = _quant_config() if self.load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_path, trust_remote_code=True, use_fast=False
        )
        self._transform = build_transform(self.INPUT_SIZE)
        self._dynamic_preprocess = dynamic_preprocess
        self._prepare_inputs = prepare_inputs_for_multimodal_embedding
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        logger.info("InternVL3 carregado: %s", self.hf_path)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        blocks = self._dynamic_preprocess(
            image, max_num=self.MAX_NUM, image_size=self.INPUT_SIZE, use_thumbnail=True
        )
        return torch.stack([self._transform(b) for b in blocks])  # [N, 3, H, W]

    def extract_image(self, image, prompt, layers=(0, -1)):
        pv = self._preprocess(image).to(torch.bfloat16)
        inp = self._prepare_inputs(self.model, self.tokenizer, pv, prompt or "")
        with torch.no_grad():
            out = self.model(
                input_ids=inp["input_ids"],
                pixel_values=inp["pixel_values"],
                image_flags=inp["image_flags"],
                output_hidden_states=True,
                return_dict=True,
            )
        return {
            li: _mean_pool_layer(out.hidden_states, li, None).numpy()
            for li in layers
        }

    def extract_text(self, text, layers=(0, -1)):
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.language_model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return {
            li: _mean_pool_layer(out.hidden_states, li, None).numpy()
            for li in layers
        }

    def generate_text(self, image, prompt, max_new_tokens=40):
        pv = self._preprocess(image).to(torch.bfloat16)
        inp = self._prepare_inputs(
            self.model, self.tokenizer, pv, prompt or "Describe this document."
        )
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inp["input_ids"],
                pixel_values=inp["pixel_values"],
                image_flags=inp["image_flags"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(
            generated[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------

class Qwen2VLExtractor(BaseExtractor):
    def load(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        quant = _quant_config() if self.load_in_4bit else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.hf_path)
        self._img_token_id: int | None = getattr(self.model.config, "image_token_id", None)
        logger.info("Qwen2.5-VL carregado: %s  img_token_id=%s", self.hf_path, self._img_token_id)

    def extract_image(self, image, prompt, layers=(0, -1)):
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": image}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        img_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=img_inputs, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)

        visual_mask = None
        if self._img_token_id is not None:
            visual_mask = (inputs.input_ids == self._img_token_id)
            if not visual_mask.any():
                visual_mask = None

        return {
            li: _mean_pool_layer(out.hidden_states, li, visual_mask).numpy()
            for li in layers
        }

    def extract_text(self, text, layers=(0, -1)):
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return {
            li: _mean_pool_layer(out.hidden_states, li, None).numpy()
            for li in layers
        }

    def generate_text(self, image, prompt, max_new_tokens=40):
        from qwen_vl_utils import process_vision_info
        content = [{"type": "image", "image": image}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=img_inputs, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Gemma-3n (Gemma 4 ~2B multimodal)
# ---------------------------------------------------------------------------

class GemmaExtractor(BaseExtractor):
    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        quant = _quant_config() if self.load_in_4bit else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.hf_path)
        self._img_token_id: int | None = getattr(self.model.config, "image_token_id", None)
        logger.info("Gemma multimodal carregado: %s", self.hf_path)

    def extract_image(self, image, prompt, layers=(0, -1)):
        effective_prompt = prompt if prompt else "Describe this document."
        inputs = self.processor(text=effective_prompt, images=image, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)

        visual_mask = None
        if self._img_token_id is not None:
            m = (inputs.input_ids == self._img_token_id)
            if m.any():
                visual_mask = m

        return {
            li: _mean_pool_layer(out.hidden_states, li, visual_mask).numpy()
            for li in layers
        }

    def extract_text(self, text, layers=(0, -1)):
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return {
            li: _mean_pool_layer(out.hidden_states, li, None).numpy()
            for li in layers
        }

    def generate_text(self, image, prompt, max_new_tokens=40):
        effective_prompt = prompt if prompt else "Describe this document."
        inputs = self.processor(text=effective_prompt, images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Ministral 3 3B (Mistral multimodal — Mistral3ForConditionalGeneration)
# ---------------------------------------------------------------------------

class Ministral3Extractor(BaseExtractor):
    def load(self):
        from transformers import Mistral3ForConditionalGeneration, AutoProcessor

        quant = _quant_config() if self.load_in_4bit else None
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.hf_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.hf_path)
        self._img_token_id: int | None = getattr(self.model.config, "image_token_id", None)
        logger.info("Ministral3 carregado: %s  img_token_id=%s", self.hf_path, self._img_token_id)

    def extract_image(self, image, prompt, layers=(0, -1)):
        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt or ""},
        ]}]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)

        visual_mask = None
        if self._img_token_id is not None:
            m = (inputs.input_ids == self._img_token_id)
            if m.any():
                visual_mask = m

        return {
            li: _mean_pool_layer(out.hidden_states, li, visual_mask).numpy()
            for li in layers
        }

    def extract_text(self, text, layers=(0, -1)):
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return {
            li: _mean_pool_layer(out.hidden_states, li, None).numpy()
            for li in layers
        }

    def generate_text(self, image, prompt, max_new_tokens=40):
        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt or ""},
        ]}]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Registro e factory
# ---------------------------------------------------------------------------

_FAMILY_MAP: dict[str, type[BaseExtractor]] = {
    "internvl":  InternVLExtractor,
    "qwen2vl":   Qwen2VLExtractor,
    "gemma":     GemmaExtractor,
    "ministral": Ministral3Extractor,
}


def build_extractor(model_cfg: dict, load_in_4bit: bool = True, device: str = "cuda") -> BaseExtractor:
    family = model_cfg["family"]
    if family not in _FAMILY_MAP:
        raise ValueError(f"Família '{family}' não reconhecida. Disponíveis: {list(_FAMILY_MAP)}")
    ext = _FAMILY_MAP[family](
        hf_path=model_cfg["hf_path"],
        load_in_4bit=load_in_4bit,
        device=device,
    )
    ext.load()
    return ext
