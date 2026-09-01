"""QLoRA supervised fine-tuning for 10-K structured extraction.

    python -m src.train.sft --smoke                       # tiny model, few steps, no GPU needed
    python -m src.train.sft                               # full run from configs/sft_llama31_8b.yaml
    python -m src.train.sft --config configs/other.yaml --set lora.r=16 training.learning_rate=1e-4

The GPU is rented by the hour, so the expensive failures are the ones that only show up after the
model has loaded: a bad data path, a missing token, an OOM at the first long batch. `--smoke` runs
the identical code path end to end on a small ungated model so those surface locally for free.

Unsloth is used when it imports (CUDA only); otherwise this falls back to plain transformers+peft,
which is what makes the smoke run possible on a Mac.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.train.data import PaddingCollator, build_dataset

logger = logging.getLogger("sft")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "sft_llama31_8b.yaml"
OUTPUT_ROOT = ROOT / "outputs"

# Ungated stand-in for the smoke run: same chat-template and tokenizer machinery, small enough to
# train on CPU/MPS in under a minute.
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_config(path: Path, overrides: list[str] | None = None) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    for override in overrides or []:
        key, _, raw = override.partition("=")
        if not raw:
            raise SystemExit(f"--set expects key=value, got {override!r}")
        node = config
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)  # keeps ints/floats/bools typed
    return config


def load_model_and_tokenizer(config: dict, token: str | None):
    """Returns (model, tokenizer). Prefers Unsloth, falls back to transformers+peft."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model_id"]
    lora = config["lora"]
    use_4bit = config.get("load_in_4bit", True) and torch.cuda.is_available()

    try:
        if not torch.cuda.is_available():
            raise ImportError("unsloth requires CUDA")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=config["max_seq_len"],
            load_in_4bit=use_4bit,
            token=token,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=config["training"].get("seed", 42),
        )
        logger.info("Loaded %s via Unsloth (4-bit=%s)", model_id, use_4bit)
        return model, tokenizer
    except ImportError as e:
        logger.info("Unsloth unavailable (%s); using transformers+peft", e)

    kwargs: dict = {"dtype": torch.bfloat16}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, **kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    logger.info("Loaded %s via transformers+peft (4-bit=%s)", model_id, use_4bit)
    return model, tokenizer


def run(config: dict, smoke: bool = False, output_dir: Path | None = None,
        resume: bool = False) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or None

    if smoke:
        # 4096 is small enough to train on CPU/MPS quickly but still clears the longest target
        # (~1,400 tokens); a window that cannot hold the target is a config error, not a test.
        config = {**config, "model_id": SMOKE_MODEL, "max_seq_len": 4096, "load_in_4bit": False}
        config["run_name"] = "smoke"
        config["wandb"] = {"enabled": False}
        config["hub"] = {"enabled": False}  # a throwaway 0.5B adapter has no business on the Hub
        config["training"] = {
            **config["training"],
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 1,
            "bf16": False,
            "optim": "adamw_torch",
            "eval_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }

    out = Path(output_dir or OUTPUT_ROOT / config["run_name"])
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config, token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = config["max_seq_len"]
    train_ds, train_stats = build_dataset(tokenizer, "train", max_seq_len, limit=8 if smoke else None)
    eval_ds = None if smoke else build_dataset(tokenizer, "val", max_seq_len)[0]

    if train_stats.truncated:
        # Truncation costs input the labels were derived from; worth seeing in the run log rather
        # than discovering it during error analysis.
        logger.warning(
            "%d/%d training examples exceeded max_seq_len=%d and lost prompt context",
            train_stats.truncated, train_stats.n, max_seq_len,
        )

    wandb_enabled = config.get("wandb", {}).get("enabled") and os.environ.get("WANDB_API_KEY")
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", config["wandb"].get("project", "finetuning-edgar"))

    training = dict(config["training"])
    if not torch.cuda.is_available():
        training["bf16"] = False  # bf16 autocast is CUDA-only in this trainer path

    hub = config.get("hub") or {}
    hub_enabled = bool(hub.get("enabled")) and bool(token)
    if hub.get("enabled") and not token:
        # Silently training with no durable destination is the failure this block exists to prevent,
        # so it is worth a loud line in the log rather than a surprise at the end of a paid run.
        logger.warning(
            "hub.enabled is set but HF_TOKEN is empty -- checkpoints stay on local disk only"
        )
    if hub_enabled:
        training.update(
            push_to_hub=True,
            hub_model_id=hub["model_id"],
            hub_strategy=hub.get("strategy", "checkpoint"),
            hub_private_repo=bool(hub.get("private", True)),
            hub_token=token,
        )
        logger.info("Checkpoints push to https://huggingface.co/%s", hub["model_id"])

    args = TrainingArguments(
        output_dir=str(out),
        run_name=config["run_name"],
        report_to=["wandb"] if wandb_enabled else [],
        remove_unused_columns=False,  # our columns are already tensors, not model kwargs
        **training,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PaddingCollator(tokenizer),
    )
    result = trainer.train(resume_from_checkpoint=resume or None)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    # The resolved config travels with the adapter: a checkpoint whose hyperparameters are only in
    # a shell history is not reproducible.
    (out / "train_config.json").write_text(json.dumps(config, indent=2, default=str))
    (out / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2))

    if hub_enabled:
        # Explicit, and last: pushes output_dir as it now stands, so train_config.json and
        # train_metrics.json land beside the adapter rather than only the weights.
        trainer.push_to_hub(commit_message=f"{config['run_name']}: final adapter")
        logger.info("Pushed final adapter to %s", hub["model_id"])

    logger.info("Saved adapter to %s", out)
    print(f"\nTrain: {train_stats.summary()}")
    print(f"Metrics: {json.dumps(result.metrics, indent=2, default=str)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", dest="overrides", default=[],
                        help="Override config values, e.g. --set lora.r=16 training.learning_rate=1e-4")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny ungated model, 8 examples, 1 epoch -- validates the pipeline")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint in --output-dir (or pull it from the "
                             "Hub first) after an interrupted run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    run(load_config(args.config, args.overrides), smoke=args.smoke,
        output_dir=args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
