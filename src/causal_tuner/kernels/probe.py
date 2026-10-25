"""Small forward/backward probes before allocating production model state."""

import torch


def probe_backends(args, device):
    from .attention import flash_dense
    from .reference import dense_attention as dense_reference
    # Keep the training RNG untouched by compilation and numerical probes.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(19)
        q = torch.randn(1, 65, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 65, 1, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        spans = ((0, 0, 65),)
        upstream = torch.randn_like(q)
        from .core_cuda import RMS
        from ..modeling.config import ModelConfig
        from ..modeling.rotary import rotary_values, rotate
        x = torch.randn(7, 4096, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.ones(4096, device=device, dtype=torch.bfloat16)
        normalized = RMS.apply(x, w, 1e-6)
        reference = (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)).bfloat16()
        torch.testing.assert_close(normalized, reference, rtol=.03, atol=.03)
        go = torch.randn_like(x)
        torch.testing.assert_close(torch.autograd.grad(normalized, x, go)[0], torch.autograd.grad(reference, x, go)[0], rtol=.03, atol=.03)
        cosine, sine = rotary_values(torch.arange(65, device=device)[None]*30000, ModelConfig(), q.dtype)
        rotated = rotate(q, cosine, sine)
        c, s = cosine.unsqueeze(-2), sine.unsqueeze(-2)
        even, odd = q[..., :64:2], q[..., 1:64:2]
        reference = torch.cat((torch.stack((even*c-odd*s, odd*c+even*s), -1).flatten(-2), q[..., 64:]), -1)
        torch.testing.assert_close(rotated, reference, rtol=.03, atol=.03)
        torch.testing.assert_close(torch.autograd.grad(rotated, q, upstream)[0], torch.autograd.grad(reference, q, upstream)[0], rtol=.03, atol=.03)
        from .core_cuda import rotate_owned_qk
        owned_q, owned_k = q.detach().clone().flatten(-2), k.detach().clone().flatten(-2)
        rotate_owned_qk(owned_q, owned_k, cosine, sine, 128)
        torch.testing.assert_close(owned_q.reshape_as(q), rotate(q, cosine, sine), rtol=0, atol=0)
        torch.testing.assert_close(owned_k.reshape_as(k), rotate(k, cosine, sine), rtol=0, atol=0)
        from ..modeling.model import MLP, RMSNorm, Runtime
        from .loss import projection_loss
        mlp = MLP(ModelConfig.tiny(), Runtime(rank=4, mlp_tile=5), device)
        for name, parameter in mlp.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=.1)
        if args.quantization == "nf4":
            for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                projection.install_nf4(projection.weight.detach(), device)
        inputs = torch.randn(2, 7, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        if args.lora and args.dtype == "bfloat16":
            norm = RMSNorm(32, 1e-6, torch.bfloat16, device)
            norm.weight.requires_grad_(False)
            signal = go.flatten()[:inputs.numel()].reshape_as(inputs)
            exact = torch.autograd.grad(norm(inputs), inputs, signal)[0].float()
            approximate = torch.autograd.grad(norm(inputs, reuse_grad=True), inputs, signal.clone())[0].float()
            if (approximate-exact).norm() > .005*exact.norm():
                raise RuntimeError("Reconstructed RMSNorm probe exceeded the gradient error tolerance")
        def plain(module, value):
            linear = torch.nn.functional.linear
            return linear(value, module.base_weight().bfloat16())+linear(linear(value, module.lora_A.bfloat16()), module.lora_B.bfloat16())
        reference = plain(mlp.down_proj, torch.nn.functional.silu(plain(mlp.gate_proj, inputs))*plain(mlp.up_proj, inputs))
        actual = mlp(inputs)
        torch.testing.assert_close(actual, reference, rtol=.05, atol=.005)
        signal = torch.randn_like(actual)
        parameters = (inputs, *(p for p in mlp.parameters() if p.requires_grad))
        for a, e in zip(torch.autograd.grad(actual, parameters, signal), torch.autograd.grad(reference, parameters, signal)):
            torch.testing.assert_close(a, e, rtol=.05, atol=.005)
        loss_head = torch.randn(4099, 32, device=device, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(4099, (2, 7), device=device)
        weights = torch.rand(2, 7, device=device)
        actual = projection_loss(inputs, loss_head, targets, weights, token_budget=5)
        logits = torch.nn.functional.linear(inputs, loss_head.bfloat16()).float()
        reference = (torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")*weights.flatten()).sum()
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
        for a, e in zip(torch.autograd.grad(actual, (inputs, loss_head)), torch.autograd.grad(reference, (inputs, loss_head))):
            torch.testing.assert_close(a, e, rtol=.05, atol=.01)
        checks = []
        if args.attn_mechanism != "vanilla":
            expected = dense_reference(q, k, v, torch.ones(1, 65, device=device))
            checks.append((flash_dense(q, k, v, spans), expected))
            packed_segments = torch.ones(1, 65, device=device)
            packed_segments[:, 17:] = 2
            checks.append((flash_dense(q, k, v, ((0, 0, 17), (0, 17, 65))),
                           dense_reference(q, k, v, packed_segments)))
            if args.maskgit_probability:
                end = torch.arange(65, device=device)[None].clone()
                end[:, 17:41] = 40
                checks.append((flash_dense(q, k, v, spans, end),
                               dense_reference(q, k, v, torch.ones(1, 65, device=device), end)))
        for actual, expected in checks:
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.03)
            ag = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
            eg = torch.autograd.grad(expected, (q, k, v), upstream, retain_graph=True)
            for a, e in zip(ag, eg):
                torch.testing.assert_close(a, e, rtol=.05, atol=.05)
        if args.loss_implementation == "cut":
            hidden = q[:, :, 0]
            head = torch.randn(257, 128, device=device, dtype=torch.bfloat16)
            targets = torch.randint(257, (1, 65), device=device)
            weights = torch.rand(1, 65, device=device)
            actual = projection_loss(hidden, head, targets, weights, implementation="cut", cut_chunk_size=17)
            expected = (torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden.float(), head.float()).flatten(0, 1),
                        targets.flatten(), reduction="none")*weights.flatten()).sum()
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.03)
            torch.testing.assert_close(torch.autograd.grad(actual, hidden)[0], torch.autograd.grad(expected, hidden)[0], rtol=.05, atol=.05)
            print("CCE controls its internal tiling; loss_token_budget does not constrain the cut backend.")
    print("Backend forward/backward probes passed. Capacity and throughput still require full-step measurement.")
