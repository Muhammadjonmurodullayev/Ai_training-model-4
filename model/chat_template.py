"""
ChatML template — same format as OpenAI/Qwen.

Each turn:
  <|im_start|>role
  content<|im_end|>

Generation prompt ends with:
  <|im_start|>assistant\n

Stop token: <|im_end|>
"""

from typing import List, Dict, Optional


# Special tokens (must be added to SentencePiece vocab as user_defined_symbols)
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
VALID_ROLES = {ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT}

DEFAULT_SYSTEM_PROMPT = (
    "Siz AI Coding Platform yordamchisisiz. Foydalanuvchi bilan o'zbek va ingliz "
    "tillarida tabiiy va do'stona suhbat qiling. Qisqa va aniq javob bering. "
    "Agar kod yozish so'ralsa, kerakli tilda toza kod yozing.\n"
    "You are the AI Coding Platform assistant. Chat naturally with the user "
    "in Uzbek and English. Be concise and friendly."
)


def render_chatml(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    system: Optional[str] = None,
) -> str:
    """
    Render a list of {role, content} messages into ChatML string.

    Example output:
        <|im_start|>system
        You are helpful.<|im_end|>
        <|im_start|>user
        Salom<|im_end|>
        <|im_start|>assistant

    Args:
        messages: list of {"role": "user"|"assistant"|"system", "content": str}
        add_generation_prompt: if True, append "<|im_start|>assistant\\n" at end
        system: optional system prompt to prepend (only if no system msg in messages)
    """
    parts: List[str] = []

    has_system = any(m.get("role") == ROLE_SYSTEM for m in messages)
    if not has_system:
        sys_prompt = system if system is not None else DEFAULT_SYSTEM_PROMPT
        if sys_prompt:
            parts.append(f"<|im_start|>{ROLE_SYSTEM}\n{sys_prompt}<|im_end|>")

    for m in messages:
        role = m.get("role", "").strip().lower()
        content = m.get("content", "")
        if role not in VALID_ROLES:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    text = "\n".join(parts)
    if add_generation_prompt:
        text += f"\n<|im_start|>{ROLE_ASSISTANT}\n"
    return text


def parse_chatml(text: str) -> List[Dict[str, str]]:
    """
    Parse a ChatML-rendered string back into messages.
    Tolerates incomplete final turn (no <|im_end|>).
    """
    import re
    pattern = re.compile(
        r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)",
        re.DOTALL,
    )
    out: List[Dict[str, str]] = []
    for m in pattern.finditer(text):
        out.append({"role": m.group(1), "content": m.group(2).strip()})
    return out


def extract_assistant_reply(generated_text: str) -> str:
    """
    Given full generated text (prompt + model output),
    extract just the last assistant turn's content.
    """
    # Find last assistant marker
    marker = "<|im_start|>assistant\n"
    idx = generated_text.rfind(marker)
    if idx == -1:
        return generated_text.strip()
    reply = generated_text[idx + len(marker):]
    # Cut at first <|im_end|>
    end = reply.find("<|im_end|>")
    if end != -1:
        reply = reply[:end]
    # Also cut if model started another turn
    for stop in ("<|im_start|>", "<|endoftext|>"):
        s = reply.find(stop)
        if s != -1:
            reply = reply[:s]
    return reply.strip()
