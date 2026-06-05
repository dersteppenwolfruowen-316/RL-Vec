"""Qwen 视觉-语言模型封装。

提供 QwenVLModel 类，支持 Qwen2.5-VL 和 Qwen3-VL 系列模型的加载、
LoRA 适配、推理和训练。
"""

from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoProcessor


def _resolve_vlm_class(model_name: str):
    """根据模型名称自动选择正确的模型类。

    Args:
        model_name: HuggingFace 模型名称。

    Returns:
        对应的模型类。
    """
    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires a transformers build that provides "
                "Qwen3VLForConditionalGeneration. Install/upgrade transformers "
                "before loading this model."
            ) from exc
        return Qwen3VLForConditionalGeneration
    if "qwen2_5" in name_lower or "qwen2.5" in name_lower:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL requires a transformers build that provides "
                "Qwen2_5_VLForConditionalGeneration. Install/upgrade transformers "
                "before loading this model."
            ) from exc
        return Qwen2_5_VLForConditionalGeneration
    from transformers import AutoModelForVision2Seq
    return AutoModelForVision2Seq


class QwenVLModel:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        device: str = "cuda",
        quantization: Optional[str] = None,
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        require_flash_attention: bool = False,
    ):
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        self.device = device
        self.model_name = base_model_name
        self.flash_attention_available = False

        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise RuntimeError(
                "QwenVLModel requires the optional training dependency `peft`. "
                "Install project requirements before loading the model: "
                "`pip install -r requirements.txt`."
            ) from exc

        print(f"Loading VLM model: {base_model_name}")

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }

        if use_flash_attention:
            if self._check_flash_attention():
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    self.flash_attention_available = True
                    print("Using Flash Attention 2")
                except Exception as e:
                    if require_flash_attention:
                        raise RuntimeError(f"Flash Attention 2 is required but failed to load: {e}")
                    print(f"Flash Attention 2 not available: {e}")
            else:
                if require_flash_attention:
                    raise RuntimeError("Flash Attention 2 is required but not supported on this device")
                print("Flash Attention 2 not supported, using default attention")

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization")
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization")

        self.model = _resolve_vlm_class(base_model_name).from_pretrained(
            base_model_name,
            **load_kwargs
        )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.processor = AutoProcessor.from_pretrained(
            base_model_name, use_fast=False
        )

        self._setup_generation_config()

    def _check_flash_attention(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def _setup_generation_config(self):
        self.default_gen_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }

    def generate(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })

        generation_kwargs.update(kwargs)

        outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def generate_svg(
        self,
        image: Any,
        max_new_tokens: int = 1024,
        svg_format: str = "detailed",
        preserve_structure: bool = True,
        include_dimensions: bool = True,
        **kwargs
    ) -> str:
        prompts = {
            "simple": """Convert this engineering drawing to SVG format.
Requirements:
- Output ONLY the SVG code, no explanations
- Use proper SVG XML syntax
- Keep all structural lines
""",
            "detailed": """You are an expert at converting engineering drawings to SVG format.

Analyze this drawing carefully and convert it to a precise SVG vector graphic.

Requirements:
1. Output ONLY the SVG code, no explanations or markdown
2. Preserve ALL structural lines accurately
3. Use proper SVG XML syntax with correct namespace
4. Main members: main structural elements
5. Secondary members: cross-bracing, diagonal supports
6. Maintain exact proportions and angles
7. Use black stroke for lines, white or transparent fill
8. Include viewBox for scalability

Example structure:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 WIDTH HEIGHT">
  <!-- Main members -->
  <line x1="..." y1="..." x2="..." y2="..." stroke="black" stroke-width="2"/>
  <!-- Secondary members -->
  <line ... />
</svg>
""",
            "cad": """You are a CAD vectorization expert.

Convert this engineering/technical drawing to an editable SVG format suitable for CAD import.

Requirements:
1. Output ONLY the raw SVG code, no markdown or explanation
2. Group elements logically (<g> tags) for CAD layer compatibility
3. Preserve ALL geometric relationships
4. Use clean, minimal paths
5. Support CAD software import (AutoCAD, FreeCAD, etc.)
6. Include dimension annotations if visible
7. Maintain engineering precision

Layer naming convention:
- Main structure → id="main_members"
- Cross bracing → id="secondary_members"
- Dimensions → id="dimensions"
- Annotations → id="annotations"
""",
            "preservation": """Analyze this technical drawing and create a highly accurate SVG representation.

Priority: Preserve ALL original information exactly.

Steps:
1. Identify all structural elements
2. Note all dimensions and annotations
3. Preserve all angles and proportions
4. Convert to clean SVG

Output ONLY the SVG code.
"""
        }

        prompt = prompts.get(svg_format, prompts["detailed"])

        if preserve_structure:
            prompt += "\nCRITICAL: Do not simplify or omit any structural elements."

        if include_dimensions:
            prompt += "\nInclude dimension lines and measurements if present in the drawing."

        return self.generate(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.8),
            **kwargs
        )

    def generate_batch(
        self,
        images: List[Any],
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        if prompts is None:
            prompts = ["Describe this image."] * len(images)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
            for img, prompt in zip(images, prompts)
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        responses = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return responses

    def visual_coding_analysis(
        self,
        image: Any,
        analysis_type: str = "full",
    ) -> Dict[str, Any]:
        analysis_prompts = {
            "structure": """Analyze the structural elements in this engineering drawing.

Identify and describe:
1. Main structural members (primary load-bearing elements)
2. Secondary members (bracing, supports)
3. Connection points and joints
4. Overall structural system

Output a structured analysis.""",

            "geometry": """Extract geometric information from this drawing.

Calculate or estimate:
1. Overall dimensions (width, height)
2. Angles between members
3. Spacing and pitch
4. Scale if indicated

Provide precise measurements.""",

            "components": """Identify all components in this technical drawing.

List each distinct element with:
1. Type (line, arc, dimension, annotation)
2. Position or coordinates
3. Dimensions or measurements
4. Purpose or function

Be comprehensive and detailed.""",

            "full": """Perform a complete technical analysis of this engineering drawing.

Cover:
1. Overall description and purpose
2. Structural system and components
3. Dimensions and measurements
4. Materials or specifications (if visible)
5. Technical annotations
6. Manufacturing or construction notes

Provide a comprehensive technical report."""
        }

        prompt = analysis_prompts.get(analysis_type, analysis_prompts["full"])

        response = self.generate(
            image,
            prompt,
            max_new_tokens=1024,
        )

        return {
            "analysis_type": analysis_type,
            "response": response,
        }

    def extract_svg_elements(
        self,
        image: Any,
        element_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if element_types is None:
            element_types = ["lines", "shapes", "dimensions", "annotations"]

        type_prompts = {
            "lines": "Extract all line elements with coordinates",
            "shapes": "Identify all geometric shapes",
            "dimensions": "Find all dimension lines and measurements",
            "annotations": "List all text annotations and labels",
        }

        elements = {}
        for etype in element_types:
            prompt = f"Extract {type_prompts.get(etype, etype)} from this drawing. Be specific about coordinates and measurements."
            response = self.generate(image, prompt, max_new_tokens=512)
            elements[etype] = response

        return elements

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": "Qwen3-VL",
            "flash_attention": self.flash_attention_available,
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.model.device),
        }
