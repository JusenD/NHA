import random

import torch

from nha_fla.models.nha.configuration_nha import NHAConfig
from nha_fla.models.nha.modeling_nha import NHAModel


def check_nans_and_inf(tensor, name):
    assert not torch.isnan(tensor).any(), f"NaNs found in {name}"
    assert not torch.isinf(tensor).any(), f"Infs found in {name}"


def test():
    # slightly larger config to put real pressure on the kernels
    config = NHAConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_heads=8,
        num_kv_heads=4,
    )
    model = NHAModel(config).to(torch.bfloat16).cuda()
    model.train()

    def run_tests(name, seqlens, batch_mode=False):
        print(f"--- Running {name} ---")
        if batch_mode:
            batch_size = len(seqlens)
            seq_len = seqlens[0]
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
            kwargs = {"input_ids": input_ids}
        else:
            total_seq_len = sum(seqlens)
            input_ids = torch.randint(0, config.vocab_size, (1, total_seq_len)).cuda()
            cu_seqlens = torch.tensor([0] + seqlens).cumsum(dim=0).to(torch.int32).cuda()
            max_seqlen = max(seqlens)
            kwargs = {"input_ids": input_ids, "cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}

        model.zero_grad()

        out = model(**kwargs)[0]
        check_nans_and_inf(out, f"{name} Output")

        loss = out.float().sum()
        check_nans_and_inf(loss, f"{name} Loss")
        loss.backward()

        has_grad = False
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                check_nans_and_inf(p.grad, f"{name} Gradients")
                grad_norm += p.grad.norm().item()

        assert has_grad, f"No gradients produced in {name}"
        print(f"{name} passed! Total seq len: {input_ids.shape[-1]}, Grad norm: {grad_norm:.4f}")

    # plain batch mode
    run_tests("Normal Batch", [128] * 4, batch_mode=True)

    # plain varlen
    run_tests("Normal Varlen", [128, 64, 200, 10])

    # stress: very long sequences
    run_tests("Stress Varlen - Long Seq", [4096, 2048, 1, 1024, 2])

    # stress: many tiny sequences (extreme batch size)
    random.seed(0)
    run_tests("Stress Varlen - Many Short", [random.randint(1, 100) for _ in range(256)])

    # stress: extreme mixed length distribution
    run_tests("Stress Varlen - Extreme Mixed", [1, 2000, 3, 50, 4, 3000, 2, 7])

    print("All varlen stress tests passed.")


if __name__ == "__main__":
    test()
