import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def target_prediction_positions(labels):
    """Logit position i predicts label position i + 1."""
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("target-only logits currently require batch_size=1")
    return torch.nonzero(labels[0, 1:] != IGNORE_INDEX, as_tuple=False).flatten()


def shifted_target_ids(labels):
    shifted = labels[:, 1:]
    return shifted[shifted != IGNORE_INDEX]


def extract_target_logits(logits, labels):
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    return shifted_logits[shifted_labels != IGNORE_INDEX]


def forward_kl(teacher_target_logits, student_target_logits, token_chunk_size=128):
    if teacher_target_logits.shape != student_target_logits.shape:
        raise AssertionError(
            f"target logits differ: teacher={teacher_target_logits.shape}, "
            f"student={student_target_logits.shape}"
        )
    teacher_target_logits = teacher_target_logits.reshape(-1, teacher_target_logits.shape[-1])
    student_target_logits = student_target_logits.reshape(-1, student_target_logits.shape[-1])
    total_kl = student_target_logits.new_zeros((), dtype=torch.float32)
    token_count = teacher_target_logits.shape[0]
    for start in range(0, token_count, token_chunk_size):
        end = min(start + token_chunk_size, token_count)
        teacher_logits = teacher_target_logits[start:end].to(
            student_target_logits.device, dtype=torch.float32
        )
        student_logits = student_target_logits[start:end].float()
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        total_kl = total_kl + (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum()
    return total_kl / token_count
