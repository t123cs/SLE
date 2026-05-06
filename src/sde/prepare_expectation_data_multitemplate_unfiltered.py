import os
import json
import math
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATES = [
    {
        "id": "main_query",
        "name": "Main Search Query",
        "instruction": (
            "Given the following document snippet, predict the Google search query "
            "that a user would type to find this content.\n"
            "Output only the search query.\n"
            "Constraints:\n"
            "- Length: 3 to 12 words.\n"
            "- Format: Natural, concise query.\n"
            "- Do not use meta-terms like 'document', 'summary', or 'snippet'.\n"
            "- Focus on the main specific object or event described.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "problem_finding",
        "name": "Problem-Oriented Query",
        "instruction": (
            "Read the snippet and write a realistic search query that someone with the same "
            "problem, confusion, or need would use.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 4 to 14 words.\n"
            "- Sound like a real web search.\n"
            "- Prioritize the user's likely intent over document phrasing.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "keyword_heavy",
        "name": "Keyword-Heavy Query",
        "instruction": (
            "Convert the document snippet into a strong keyword-style search query for retrieval.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 3 to 12 words.\n"
            "- Emphasize rare, identifying, content-bearing words.\n"
            "- Avoid filler words unless necessary.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "entity_event",
        "name": "Entity or Event Query",
        "instruction": (
            "Write a search query focusing on the main entity, event, case, product, law, or concept "
            "that best identifies this snippet.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 3 to 10 words.\n"
            "- Make it precise and disambiguating.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "how_why",
        "name": "How or Why Query",
        "instruction": (
            "Imagine a user wants to understand or resolve what this snippet is about.\n"
            "Write a plausible 'how', 'why', 'can', or explanatory search query.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 4 to 14 words.\n"
            "- Keep it natural and specific.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "alt_paraphrase",
        "name": "Alternate Paraphrase Query",
        "instruction": (
            "Write an alternative search query that could retrieve this snippet using different wording "
            "from the original text.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 3 to 12 words.\n"
            "- Prefer paraphrases, synonyms, or reformulations.\n"
            "- Still keep the query realistic.\n\n"
            "Snippet: {doc_text}"
        ),
    },
]


def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) >= 2:
                corpus[parts[0]] = parts[1]
    return corpus


def build_prompt(tokenizer, template, doc_text, max_doc_chars):
    user_text = template["instruction"].format(doc_text=doc_text[:max_doc_chars])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def parse_template_ids(arg):
    if arg.strip().lower() == "all":
        return [t["id"] for t in PROMPT_TEMPLATES]
    return [x.strip() for x in arg.split(",") if x.strip()]


def main(args):
    print(f"[*] Loading generation model: {args.model_path}")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Local model directory not found: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    print("[*] Loading corpus...")
    corpus = load_corpus(args.corpus_path)
    all_doc_ids = sorted(corpus.keys())
    start = max(0, args.start_idx)
    end = len(all_doc_ids) if args.max_docs <= 0 else min(len(all_doc_ids), start + args.max_docs)
    target_ids = all_doc_ids[start:end]
    print(f"[*] Corpus docs: {len(all_doc_ids)} | selected: {len(target_ids)} (start={start}, end={end})")

    selected_template_ids = set(parse_template_ids(args.template_ids))
    templates = [t for t in PROMPT_TEMPLATES if t["id"] in selected_template_ids]
    if not templates:
        raise ValueError(f"No templates matched template_ids={args.template_ids}")
    print(f"[*] Templates: {[t['id'] for t in templates]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    meta_path = args.output_path + ".meta.json"

    quote_ids = set()
    for s in ['"', ' "', "\u201c", "\u201d"]:
        quote_ids.update(tokenizer.encode(s, add_special_tokens=False))
    eos_ids = {tokenizer.eos_token_id}
    for special in ["<|eot_id|>", "<|end_of_text|>"]:
        eos_ids.update(tokenizer.encode(special, add_special_tokens=False))

    tasks = []
    for doc_id in target_ids:
        doc_text = corpus[doc_id]
        for template in templates:
            for sample_idx in range(args.samples_per_template):
                tasks.append({
                    "doc_id": doc_id,
                    "doc_text": doc_text,
                    "template": template,
                    "sample_idx": sample_idx,
                })

    total_tasks = len(tasks)
    print(
        f"[*] Planned generations: docs={len(target_ids)} x templates={len(templates)} "
        f"x samples={args.samples_per_template} = {total_tasks}"
    )

    do_sample = args.do_sample
    if args.samples_per_template > 1 and not do_sample:
        print("[*] samples_per_template > 1 detected; forcing do_sample=True for diversity.")
        do_sample = True

    generation_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    print(f"[*] Generation config: {generation_cfg}")

    written = 0
    batch_size = args.infer_batch_size

    with open(args.output_path, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=total_tasks, desc="Generate")

        for batch in chunked(tasks, batch_size):
            prompts = [
                build_prompt(tokenizer, item["template"], item["doc_text"], args.max_doc_chars)
                for item in batch
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to("cuda")

            input_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                gen_out = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    top_k=args.top_k if do_sample else None,
                    repetition_penalty=args.repetition_penalty,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=list(eos_ids),
                )

            generated = gen_out.sequences[:, input_len:]
            topk_indices_all = [[] for _ in range(len(batch))]
            topk_probs_all = [[] for _ in range(len(batch))]
            generated_token_ids = [[] for _ in range(len(batch))]

            for b in range(len(batch)):
                for t, score_t in enumerate(gen_out.scores):
                    tok = generated[b, t].item()
                    if tok == tokenizer.pad_token_id and t > 0:
                        break
                    p_dist = F.softmax(score_t[b].float(), dim=-1)
                    top_p_t, top_i_t = torch.topk(p_dist, k=args.probs_topk)
                    topk_indices_all[b].append(top_i_t.tolist())
                    topk_probs_all[b].append(top_p_t.tolist())
                    generated_token_ids[b].append(tok)
                    if tok in eos_ids:
                        break

            for i, item in enumerate(batch):
                seq = generated_token_ids[i]
                indices = topk_indices_all[i]
                probs = topk_probs_all[i]

                q_text = tokenizer.decode(seq, skip_special_tokens=True).strip().strip('"\'')

                while indices and indices[0] and indices[0][0] in quote_ids:
                    indices.pop(0)
                    probs.pop(0)
                while indices and indices[-1] and indices[-1][0] in eos_ids:
                    indices.pop()
                    probs.pop()
                while indices and indices[-1] and indices[-1][0] in quote_ids:
                    indices.pop()
                    probs.pop()

                record = {
                    "doc_id": item["doc_id"],
                    "query_text": q_text,
                    "indices": indices,
                    "probs": probs,
                    "prompt_id": item["template"]["id"],
                    "prompt_name": item["template"]["name"],
                    "sample_idx": item["sample_idx"],
                    "doc_char_len": len(item["doc_text"]),
                    "generation": {
                        "do_sample": do_sample,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "repetition_penalty": args.repetition_penalty,
                        "max_new_tokens": args.max_new_tokens,
                        "probs_topk": args.probs_topk,
                    },
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            pbar.update(len(batch))

        pbar.close()

    meta = {
        "model_path": args.model_path,
        "corpus_path": args.corpus_path,
        "output_path": args.output_path,
        "num_corpus_docs": len(all_doc_ids),
        "num_selected_docs": len(target_ids),
        "start_idx": start,
        "end_idx": end,
        "num_templates": len(templates),
        "templates": templates,
        "samples_per_template": args.samples_per_template,
        "num_written_rows": written,
        "generation": generation_cfg,
        "max_prompt_length": args.max_prompt_length,
        "max_doc_chars": args.max_doc_chars,
        "probs_topk": args.probs_topk,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("")
    print("======================================================================")
    print("  Multi-Template Unfiltered Synthesis Completed")
    print("----------------------------------------------------------------------")
    print(f"  Output rows   : {written}")
    print(f"  Output file   : {args.output_path}")
    print(f"  Metadata file : {meta_path}")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--llama_path", dest="model_path", help=argparse.SUPPRESS)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_docs", type=int, default=-1, help="<=0 means use all docs from start_idx")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--infer_batch_size", type=int, default=24)
    parser.add_argument("--template_ids", type=str, default="all")
    parser.add_argument("--samples_per_template", type=int, default=2)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--probs_topk", type=int, default=10)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_doc_chars", type=int, default=2000)
    main(parser.parse_args())
