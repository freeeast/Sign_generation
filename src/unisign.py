"""Pose-only inference adapter around the official Uni-Sign implementations."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np


class UniSign:
    def __init__(self, repo, checkpoint, mt5, device, threads):
        import torch
        from transformers import MT5Config

        repo, checkpoint, mt5 = (Path(path).resolve(strict=True)
                                  for path in (repo, checkpoint, mt5))
        if threads:
            torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.torch = torch
        self.device = torch.device(device)
        sys.path.insert(0, str(repo))
        config = importlib.import_module("config")
        config.mt5_path = str(mt5)
        datasets = importlib.import_module("datasets")
        models = importlib.import_module("models")
        self.load_part_kp = datasets.load_part_kp
        args = SimpleNamespace(dataset="How2Sign", hidden_dim=256, rgb_support=False, label_smoothing=0.)
        self.collate = datasets.S2T_Dataset_online(args=SimpleNamespace(
            rgb_support=False, max_length=256)).collate_fn
        # Strict loading below replaces every state tensor. Only the base config and
        # tokenizer are needed locally to construct the upstream architecture.
        mt5_class = models.MT5ForConditionalGeneration
        mt5_config = MT5Config.from_pretrained(mt5, local_files_only=True)
        with patch.object(mt5_class, "from_pretrained", side_effect=lambda *_: mt5_class(mt5_config)):
            self.model = models.Uni_Sign(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval().to(self.device)

    def _encode(self, xy, scores, max_frames):
        frames = len(xy)
        if frames > max_frames:
            indices = np.linspace(0, frames - 1, max_frames, dtype=int)
            xy, scores = xy[indices], scores[indices]
        parts = self.load_part_kp(xy[:, None, ...], scores[:, None, ...])
        src, tgt = self.collate([("clip", parts, "", "", {})])
        for key, value in src.items():
            if self.torch.is_tensor(value):
                src[key] = value.to(self.device, dtype=self.torch.float32)
        out = self.model(src, tgt)
        attention = out["attention_mask"]
        encoded = self.model.mt5_model.encoder(
            inputs_embeds=out["inputs_embeds"], attention_mask=attention, return_dict=True)
        # Pose tokens are appended to the fixed language prompt by Uni-Sign.
        hidden = encoded.last_hidden_state[0, -len(xy):]
        return encoded, attention, hidden.mean(dim=0).cpu().numpy(), len(xy)

    def _score_targets(self, encoded, attention, targets, *, token_nll=False):
        if not targets:
            return ([], []) if token_nll else []
        tokenizer = self.model.mt5_tokenizer
        labels = tokenizer(targets, return_tensors="pt", padding=True, truncation=True,
                           max_length=50)["input_ids"].to(self.device)
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        hidden = encoded.last_hidden_state.expand(len(targets), -1, -1)
        output = self.model.mt5_model(
            encoder_outputs=(hidden,), attention_mask=attention.expand(len(targets), -1),
            labels=labels, return_dict=True)
        losses = self.torch.nn.functional.cross_entropy(
            output.logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
        counts = (labels != -100).sum(dim=1)
        average = (losses.sum(dim=1) / counts).cpu().tolist()
        if token_nll:
            return average, [loss[:count].cpu().tolist() for loss, count in zip(losses, counts.tolist())]
        return average

    def teacher_log_probs(self, xy, scores, target, max_frames):
        """Full vocabulary log-probabilities at the same gold-prefix positions."""
        with self.torch.inference_mode():
            encoded, attention, _, _ = self._encode(xy, scores, max_frames)
            labels = self.model.mt5_tokenizer([target], return_tensors="pt", truncation=True,
                                              max_length=50)["input_ids"].to(self.device)
            outputs = self.model.mt5_model(encoder_outputs=encoded, attention_mask=attention,
                                           labels=labels, return_dict=True)
            values = outputs.logits[0].double().cpu()
            return values - self.torch.logsumexp(values, dim=-1, keepdim=True)

    def infer(self, xy, scores, targets, *, max_frames, beams=0, max_tokens=100, token_nll=False):
        with self.torch.inference_mode():
            encoded, attention, pooled, frames = self._encode(xy, scores, max_frames)
            nll = self._score_targets(encoded, attention, targets, token_nll=token_nll)
            per_token = None
            if token_nll:
                nll, per_token = nll
            prediction = None
            if beams:
                decoded = self.model.mt5_model.generate(
                    encoder_outputs=encoded, attention_mask=attention, max_new_tokens=max_tokens,
                    num_beams=beams)
                prediction = self.model.mt5_tokenizer.batch_decode(decoded, skip_special_tokens=True)[0]
        return dict(prediction=prediction, target_nll=nll, frames=frames, embedding=pooled, token_nll=per_token)


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else None
