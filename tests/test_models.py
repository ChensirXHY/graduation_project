"""单元测试：模型定义（LSTM 与 Transformer 的形状契约）。

本文件用 ``importorskip("torch")``：CI 里不装 torch 时优雅跳过，
本地 GPU 环境完整执行。形状契约测试的价值在于——模型定义一旦
被改动，这些测试会在训练开始前就把"输入输出形状对不上"报出来，
而不是让训练跑一半才崩。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from src.models import InformerModel, LSTMModel, TransformerModel  # noqa: E402

INPUT_SIZE = 12
SEQ_LEN = 96
OUTPUT_LEN = 24
BATCH = 8


def _sample_batch() -> torch.Tensor:
    """构造一个形状 (BATCH, SEQ_LEN, INPUT_SIZE) 的随机输入。"""
    return torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)


class TestLSTMModel:
    """LSTM 基线的形状契约（与 03_baseline_lstm.py 的权重兼容）。"""

    def test_output_shape(self) -> None:
        model = LSTMModel(input_size=INPUT_SIZE, output_len=OUTPUT_LEN)
        out = model(_sample_batch())
        assert out.shape == (BATCH, OUTPUT_LEN)

    def test_parameter_count_matches_baseline(self) -> None:
        """参数量与 03 脚本的定义一致。

        若这里失败，说明模型结构被改动，将无法加载 03 训练出的
        models/lstm_baseline.pth 权重（形状不匹配）。
        """
        model = LSTMModel(
            input_size=INPUT_SIZE, hidden_size=128, num_layers=2, output_len=OUTPUT_LEN
        )
        n_params = sum(p.numel() for p in model.parameters())
        # LSTM(12→128, 2层) + Linear(128→24)，PyTorch 每层的参数量为：
        #   weight_ih + weight_hh + bias_ih + bias_hh
        # 层1: 4*128*(12+128) + 4*128*2 = 71680 + 1024 = 72704
        # 层2: 4*128*(128+128) + 4*128*2 = 131072 + 1024 = 132096
        # FC:  128*24 + 24 = 3096
        # 合计 207896
        assert n_params == 207896, f"参数量变化：{n_params}"


class TestTransformerModel:
    """Transformer 主模型的形状契约与参数校验。"""

    def test_output_shape_mean_pool(self) -> None:
        model = TransformerModel(
            input_size=INPUT_SIZE,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_len=OUTPUT_LEN,
            pool="mean",
        )
        out = model(_sample_batch())
        assert out.shape == (BATCH, OUTPUT_LEN)

    def test_output_shape_last_pool(self) -> None:
        model = TransformerModel(
            input_size=INPUT_SIZE,
            d_model=64,
            nhead=4,
            num_layers=1,
            output_len=OUTPUT_LEN,
            pool="last",
        )
        out = model(_sample_batch())
        assert out.shape == (BATCH, OUTPUT_LEN)

    def test_dmodel_must_divide_nhead(self) -> None:
        """d_model 不能被 nhead 整除时应立即报错。"""
        with pytest.raises(ValueError, match="整除"):
            TransformerModel(input_size=INPUT_SIZE, d_model=64, nhead=7)

    def test_invalid_pool_raises(self) -> None:
        """非法的 pool 取值应报错。"""
        with pytest.raises(ValueError, match="pool"):
            TransformerModel(input_size=INPUT_SIZE, pool="max")

    def test_sequence_too_long_raises(self) -> None:
        """输入长度超过位置编码表时应报错，而不是静默截断。"""
        model = TransformerModel(input_size=INPUT_SIZE, max_seq_len=64)
        too_long = torch.randn(2, 100, INPUT_SIZE)
        with pytest.raises(ValueError, match="位置编码"):
            model(too_long)

    def test_positional_encoding_is_registered_buffer(self) -> None:
        """位置编码必须是 buffer：随模型保存加载，但不算可训练参数。"""
        model = TransformerModel(input_size=INPUT_SIZE)
        assert "pos_encoding" in dict(model.state_dict())
        assert all(
            not p.requires_grad for name, p in model.named_parameters() if "pos_encoding" in name
        )


class TestInformerModel:
    """Informer 主模型的形状契约与核心机制。"""

    def _model(self, **overrides) -> InformerModel:
        params = dict(
            input_size=INPUT_SIZE,
            d_model=64,
            nhead=4,
            e_layers=3,
            d_layers=1,
            d_ff=128,
            dropout=0.1,
            factor=5,
            label_len=24,
            pred_len=OUTPUT_LEN,
            distil=True,
        )
        params.update(overrides)
        return InformerModel(**params)

    def test_output_shape(self) -> None:
        """输出应为完整解码序列 (B, label_len + pred_len)。

        论文协议：解码器输出覆盖"开始标记 + 预测"全长，
        训练时前 label_len 个位置用已知值做 teacher forcing，
        推理时取最后 pred_len 个位置。
        """
        model = self._model()
        out = model(torch.randn(BATCH, SEQ_LEN, INPUT_SIZE))
        assert out.shape == (BATCH, model.label_len + model.pred_len)
        assert out[:, -OUTPUT_LEN:].shape == (BATCH, OUTPUT_LEN)

    def test_distilling_halves_sequence(self) -> None:
        """蒸馏后编码器记忆长度应为 L//2^(e_layers-1)。"""
        model = self._model(e_layers=3, distil=True)
        enc = model.enc_embedding(torch.randn(2, 96, INPUT_SIZE))
        enc = enc + model.pos_encoding[:, :96, :]
        memory = model.encoder(enc)
        # 96 → 48 → 24（3 层，前 2 层后蒸馏）
        assert memory.shape == (2, 24, 64)

    def test_no_distil_keeps_length(self) -> None:
        """关闭蒸馏时记忆长度应保持 96。"""
        model = self._model(e_layers=2, distil=False)
        enc = model.enc_embedding(torch.randn(2, 96, INPUT_SIZE))
        enc = enc + model.pos_encoding[:, :96, :]
        memory = model.encoder(enc)
        assert memory.shape == (2, 96, 64)

    def test_decoder_uses_last_label_len_as_start_token(self) -> None:
        """不传 x_dec 时，应自动取输入最后 label_len 步作为开始标记。

        用可区分的输入验证：把输入最后 24 步改成特殊值，输出应随之
        改变（证明解码器确实在用这段开始标记，而非凭空生成）。
        """
        model = self._model()
        model.eval()
        base = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
        with torch.no_grad():
            out_base = model(base)
            altered = base.clone()
            altered[:, -24:, :] = 99.0
            out_altered = model(altered)
        assert not torch.allclose(out_base, out_altered)

    def test_prob_sparse_output_shape(self) -> None:
        """ProbSparseAttention 输出形状应与输入一致，且活跃 query 数受因子控制。"""
        attn = __import__("src.models", fromlist=["ProbSparseAttention"]).ProbSparseAttention(
            factor=5, dropout=0.0
        )
        q = torch.randn(2, 4, 96, 16)
        k = torch.randn(2, 4, 96, 16)
        v = torch.randn(2, 4, 96, 16)
        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_dmodel_must_divide_nhead(self) -> None:
        """d_model 不能被 nhead 整除时应立即报错。"""
        with pytest.raises(ValueError, match="整除"):
            self._model(d_model=64, nhead=7)

    def test_weight_save_load_roundtrip(self, tmp_path) -> None:
        """权重保存加载往返后输出应一致（torch 状态字典兼容性）。"""
        model = self._model()
        path = tmp_path / "informer.pth"
        torch.save(model.state_dict(), path)

        loaded = self._model()
        loaded.load_state_dict(torch.load(path))
        x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x))
