import json
import re


BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
IGNORE_INDEX = -100
MAX_SEQUENCE_LENGTH = 2048

SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|context|>",
    "<|answer|>",
    "<|end|>",
]

STRUCTURED_FORMATS = {"qa", "choice_qa", "instruction", "chat"}


def coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(coerce_text(item) for item in value if coerce_text(item))
    if isinstance(value, dict):
        for key in ("text", "content", "value", "answer", "output"):
            if key in value:
                return coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_whitespace(text):
    text = coerce_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_answer_text(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        if "text" in value:
            text = value["text"]
            if isinstance(text, list):
                return coerce_text(text[0] if text else "")
            return coerce_text(text)
        for key in ("answer", "answers", "output", "completion"):
            if key in value:
                return first_answer_text(value[key])
    if isinstance(value, list):
        if not value:
            return ""
        return first_answer_text(value[0])
    return coerce_text(value)


def first_present(sample, *keys):
    for key in keys:
        if key not in sample:
            continue
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def normalize_choices(value):
    if value is None:
        return []

    if isinstance(value, dict):
        for key in ("text", "choices", "labels", "translation", "choices_translation"):
            if key in value:
                return normalize_choices(value[key])
        return [normalize_whitespace(item) for item in value.values() if normalize_whitespace(item)]

    if not isinstance(value, list):
        text = normalize_whitespace(value)
        return [text] if text else []

    choices = []
    for item in value:
        if isinstance(item, dict):
            text = first_present(item, "text", "content", "value", "label", "answer", "output")
        else:
            text = item
        text = normalize_whitespace(text)
        if text:
            choices.append(text)
    return choices


def choice_index(value, choices):
    if value is None:
        return 0

    raw = coerce_text(value).strip()
    if raw.isdigit():
        return int(raw)

    if len(raw) == 1 and raw.isalpha():
        return ord(raw.upper()) - ord("A")

    match = re.match(r"^([A-Za-z])\)", raw)
    if match:
        return ord(match.group(1).upper()) - ord("A")

    raw_lower = raw.lower()
    for index, choice in enumerate(choices):
        if raw_lower == choice.lower() or raw_lower in choice.lower():
            return index

    return 0


def ensure_meta(result, **values):
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    for key, value in values.items():
        if value is not None and key not in meta:
            meta[key] = value
    if meta:
        result["meta"] = meta


def adapt_piqa_schema(result):
    question = first_present(
        result,
        "input_translation",
        "goal_translation",
        "question_translation",
        "input",
        "goal",
        "question",
        "prompt",
    )
    choices = normalize_choices(first_present(result, "choices_translation", "choices"))
    if not choices and (result.get("sol1") or result.get("sol2")):
        choices = normalize_choices([result.get("sol1", ""), result.get("sol2", "")])

    result.setdefault("source", "piqa_italian")
    result.setdefault("language", "it")
    result["format"] = "choice_qa"

    if question and choices:
        option_lines = []
        for index, choice in enumerate(choices):
            option_lines.append(f"{chr(ord('A') + index)}) {choice}")

        gold = first_present(result, "gold_index", "label", "answer", "completion", "correct")
        gold_index = choice_index(gold, choices)
        if 0 <= gold_index < len(choices):
            result.setdefault(
                "prompt",
                "Scegli l'opzione logicamente corretta e rispondi solo con la scelta migliore.\n\n"
                f"Situazione:\n{normalize_whitespace(question)}\n\nOpzioni:\n"
                + "\n".join(option_lines),
            )
            result.setdefault(
                "completion",
                f"{chr(ord('A') + gold_index)}) {choices[gold_index]}",
            )

    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    ensure_meta(
        result,
        id=result.get("id"),
        task=metadata.get("category") or result.get("task") or "physical_reasoning",
        schema_adapter="piqa_italian",
    )
    return result


def adapt_squad_schema(result):
    context = first_present(result, "context", "passage")
    question = first_present(result, "question", "query", "prompt")
    answer = first_answer_text(first_present(result, "answers", "answer", "completion", "output"))

    result.setdefault("source", "squad_it")
    result.setdefault("language", "it")
    result["format"] = "qa"

    if context and question and answer:
        result.setdefault(
            "prompt",
            "Leggi il contesto e rispondi in italiano in modo breve e preciso.\n\n"
            f"<|context|>\n{normalize_whitespace(context)}\n\nDomanda:\n{normalize_whitespace(question)}",
        )
        result.setdefault("completion", answer)

    ensure_meta(result, id=result.get("id"), task=result.get("task") or "extractive_qa", schema_adapter="squad_it")
    return result


def adapt_instruction_schema(result):
    messages = first_present(result, "messages", "conversations", "conversation")
    if messages:
        result["format"] = "chat"
        result["messages"] = messages
        return result

    instruction = first_present(result, "instruction", "prompt", "input", "question")
    output = first_present(result, "output", "response", "answer", "completion")
    extra_input = result.get("input")
    if instruction and output:
        prompt = normalize_whitespace(instruction)
        if extra_input and extra_input != instruction and "instruction" in result:
            prompt = f"{prompt}\n\nInput:\n{normalize_whitespace(extra_input)}"
        result.setdefault("prompt", prompt)
        result.setdefault("completion", output)
        result.setdefault("format", "instruction")

    return result


def adapt_stackexchange_schema(result):
    question = first_present(result, "question", "title", "prompt")
    answer = first_answer_text(first_present(result, "answers", "accepted_answer", "answer", "completion", "response"))

    result.setdefault("source", "stackexchange_auto")
    result.setdefault("language", "en")
    result["format"] = "qa"

    if question and answer:
        result.setdefault("prompt", question)
        result.setdefault("completion", answer)

    ensure_meta(result, task=result.get("task") or "automotive_qa", schema_adapter="stackexchange_auto")
    return result


def adapt_generic_structured_schema(result):
    if "messages" in result or "conversations" in result:
        result["format"] = "chat"
        return result

    if result.get("prompt") and result.get("completion"):
        result.setdefault("format", "instruction")
        return result

    if result.get("question") and (result.get("answer") or result.get("answers")):
        result.setdefault("format", "qa")
        result.setdefault("prompt", result.get("question"))
        result.setdefault("completion", first_answer_text(result.get("answer", result.get("answers"))))
        return result

    return adapt_instruction_schema(result)


def adapt_source_schema(result):
    source = coerce_text(result.get("source", "")).strip().lower()
    keys = set(result.keys())

    if source == "piqa_italian" or keys & {"input_translation", "choices_translation", "gold_index", "sol1", "sol2"}:
        return adapt_piqa_schema(result)

    if source == "squad_it" or {"context", "question", "answers"}.issubset(keys):
        return adapt_squad_schema(result)

    if source == "stackexchange_auto":
        return adapt_stackexchange_schema(result)

    if source == "evol_instruct_italian":
        result.setdefault("source", "evol_instruct_italian")
        result.setdefault("language", "it")
        return adapt_instruction_schema(result)

    if keys & {"messages", "conversations", "conversation", "instruction", "response"}:
        return adapt_instruction_schema(result)

    return adapt_generic_structured_schema(result)


def normalize_role(role):
    value = coerce_text(role).strip().lower()
    if value in {"human", "user", "utente", "question", "prompt"}:
        return "user"
    if value in {"gpt", "assistant", "assistente", "answer", "response"}:
        return "assistant"
    if value in {"system", "sistema"}:
        return "system"
    return value or "user"


def normalize_messages(messages):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return [{"role": "user", "content": messages}]

    if not isinstance(messages, list):
        return []

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            content = normalize_whitespace(message)
            if content:
                normalized.append({"role": "user", "content": content})
            continue

        role = normalize_role(message.get("role", message.get("from", "user")))
        content = normalize_whitespace(
            message.get("content", message.get("value", message.get("text", "")))
        )
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def normalize_sample(sample, source=None, default_format="text", language=None):
    if isinstance(sample, str):
        sample = {"text": sample}
    if not isinstance(sample, dict):
        sample = {"text": coerce_text(sample)}

    result = dict(sample)
    if source and not result.get("source"):
        result["source"] = source
    if language and not result.get("language"):
        result["language"] = language

    result = adapt_source_schema(result)

    sample_format = result.get("format") or result.get("type") or default_format
    result["format"] = str(sample_format).strip().lower()

    if "messages" in result or "conversations" in result:
        result["format"] = "chat"
        result["messages"] = normalize_messages(result.get("messages", result.get("conversations")))
        result.pop("conversations", None)

    if "prompt" in result:
        result["prompt"] = normalize_whitespace(result.get("prompt"))
    if "completion" in result:
        result["completion"] = normalize_whitespace(result.get("completion"))
    if "text" in result:
        result["text"] = normalize_whitespace(result.get("text"))

    return result


def is_structured_sample(sample):
    return normalize_sample(sample).get("format") in STRUCTURED_FORMATS


def render_prompt_completion(sample):
    sample = normalize_sample(sample)
    sample_format = sample.get("format", "text")

    if sample_format == "chat":
        messages = normalize_messages(sample.get("messages", []))
        if not messages:
            return "", ""

        prompt_messages = messages[:-1]
        completion_message = messages[-1]
        if completion_message.get("role") != "assistant":
            return render_training_text(sample), ""

        prompt = render_messages(prompt_messages, include_assistant_answers=True)
        completion = f"<|assistant|>\n{completion_message['content']}\n<|end|>"
        return prompt.strip(), completion.strip()

    if sample_format in {"qa", "choice_qa", "instruction"}:
        prompt = sample.get("prompt") or sample.get("instruction") or sample.get("question") or ""
        completion = sample.get("completion") or sample.get("answer") or sample.get("output") or ""
        prompt = normalize_whitespace(prompt)
        completion = normalize_whitespace(completion)
        if prompt and not prompt.startswith("<|user|>"):
            prompt = f"<|user|>\n{prompt}\n<|end|>"
        if completion and not completion.startswith("<|assistant|>"):
            completion = f"<|assistant|>\n{completion}\n<|end|>"
        return prompt.strip(), completion.strip()

    text = render_training_text(sample)
    return text, ""


def render_messages(messages, include_assistant_answers=True):
    rendered = []
    for message in normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        if role == "assistant" and not include_assistant_answers:
            continue
        token = {
            "system": "<|system|>",
            "assistant": "<|assistant|>",
            "user": "<|user|>",
        }.get(role, "<|user|>")
        rendered.append(f"{token}\n{content}\n<|end|>")
    return "\n".join(rendered).strip()


def render_training_text(sample):
    sample = normalize_sample(sample)
    sample_format = sample.get("format", "text")

    if sample_format == "chat":
        return render_messages(sample.get("messages", []), include_assistant_answers=True)

    if sample_format in {"qa", "choice_qa", "instruction"}:
        prompt, completion = render_prompt_completion(sample)
        return "\n".join(part for part in (prompt, completion) if part).strip()

    return normalize_whitespace(sample.get("text", ""))


def sample_language(sample):
    sample = normalize_sample(sample)
    return coerce_text(sample.get("language", "")).strip().lower()


def sample_source(sample):
    sample = normalize_sample(sample)
    return coerce_text(sample.get("source", "unknown")).strip() or "unknown"
