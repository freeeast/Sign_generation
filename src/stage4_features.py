"""Capture actual pretrained branch activations without changing Uni-Sign inference."""
import numpy as np

from .unisign import cosine


class FrozenFeedback:
    def __init__(self, adapter):
        self.adapter = adapter

    def extract(self, xy, scores, text=None):
        model = self.adapter
        captures = {}
        handles = []
        for name, branches in (("hand_spatial", model.model.gcn_modules),
                               ("hand_temporal", model.model.fusion_gcn_modules)):
            for side in ("left", "right"):
                def capture(module, inputs, out, key=(name, side)):
                    # Clone at the hook so downstream residuals cannot change pre-fusion values.
                    captures[key] = out.detach().clone()
                handles.append(branches[side].register_forward_hook(capture))
        try:
            with model.torch.inference_mode():
                encoded, mask, pooled, frames = model._encode(xy, scores, 256)
                if frames != len(xy):
                    raise ValueError("Stage 4 requires aligned unsampled token rows")
                out = {"global_encoder": pooled, "frame_encoder": encoded.last_hidden_state[0, -frames:].cpu().numpy()}
                for name in ("hand_spatial", "hand_temporal"):
                    side_arrays = []
                    for side in ("left", "right"):
                        data = captures[name, side]
                        if data.shape[0] != 1 or data.shape[2] != frames or data.shape[3] != 21:
                            raise ValueError(f"Unexpected hand activation shape: {data.shape}")
                        side_arrays.append(data[0].permute(1, 2, 0).flatten(1).cpu().numpy())
                    out[name] = np.stack(side_arrays, axis=1)
                out["nll"] = model._score_targets(encoded, mask, [text])[0] if text is not None else None
                return out
        finally:
            for hook in handles:
                hook.remove()


def distances(candidate, reference):
    result = {"global_encoder": 1. - cosine(candidate["global_encoder"], reference["global_encoder"])}
    for name in ("frame_encoder", "hand_spatial", "hand_temporal"):
        a, b = candidate[name], reference[name]
        same = np.sum(a * b, axis=-1)
        norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        valid = norm > 1e-8
        result[name] = float(np.mean(1 - same[valid] / norm[valid]))
    return result
