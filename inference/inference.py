"""
CURRENTLY UNUSED.

# Inference for forecasting LLMs, use huggingface implementation:


def huggingface_generate(
    model,
    tokenizer,
    message: str,
    chat_history: Optional[list[tuple[str, str]]] = [],
    system_prompt: str = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    ) -> Iterator[str]:
    conversation = []
    # if system_prompt and 'mistral' not in tokenizer.name_or_path:
    #     conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})
    if 'starling' in tokenizer.name_or_path.lower():
        chat_convo = tokenizer.apply_chat_template(conversation, tokenize=False) + 'GPT4 Correct Assistant:'
        input_ids = tokenizer(chat_convo, return_tensors='pt')['input_ids']
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors='pt')
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)

    return "".join(outputs)

"""
