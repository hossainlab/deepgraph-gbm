"""Model forward-pass shapes, finite loss, one-batch overfit sanity."""

import torch

from deepgraph_gbm.models.multitask import DeepGraphGBM, multitask_loss


def _random_graph(n=100, f=50, k=6):
    x = torch.randn(n, f)
    ei = torch.randint(0, n, (2, n * k))
    ei = ei[:, ei[0] != ei[1]]
    return x, ei


def test_forward_shapes():
    model = DeepGraphGBM(in_dim=50, hidden_dims=[32, 32, 16])
    x, ei = _random_graph()
    out = model(x, ei)
    assert out["niche_logits"].shape == (100, 4)
    assert out["mes_prob"].shape == (100,)
    assert out["risk"].dim() == 0
    assert out["z"].shape == (100, 16)


def test_loss_finite():
    model = DeepGraphGBM(in_dim=50, hidden_dims=[32, 32, 16])
    x, ei = _random_graph()
    out = model(x, ei)
    niche_y = torch.randint(0, 4, (100,))
    niche_y[:10] = -1  # excluded spots
    mes_y = torch.rand(100)
    mes_y[:5] = float("nan")
    risk_y = torch.tensor(0.3)
    loss, parts = multitask_loss(out, niche_y, mes_y, risk_y)
    assert torch.isfinite(loss)
    assert set(parts) == {"niche", "mes", "survival"}


def test_one_batch_overfit():
    torch.manual_seed(0)
    model = DeepGraphGBM(in_dim=50, hidden_dims=[64, 64, 32], dropout=0.0)
    x, ei = _random_graph(n=60, f=50)
    niche_y = torch.randint(0, 4, (60,))
    mes_y = torch.rand(60)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for step in range(300):
        out = model(x, ei, with_risk=False)
        loss, _ = multitask_loss(out, niche_y, mes_y, None, weights={"niche": 1.0, "mes": 0.5})
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2  # loss drops by >80%
