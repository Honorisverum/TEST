"""
=================================================
                TRANSFORMER MODEL
            ADAPTING FOR DEEP RL ALGORITHM
=================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MakeTransformer(nn.Module):

    def __init__(self, d_model, n_frames, n_gts, bb_dim):
        super(MakeTransformer, self).__init__()

        self.d_model = d_model
        self.d_model_ = d_model + bb_dim
        self.n_frames = n_frames
        self.n_gts = n_gts
        self.bb_dim = bb_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.use_gpu = torch.cuda.is_available()
        self.hh = 4
        assert self.d_model_ % self.hh == 0, "incorrect inner model dim"

        # i
        self.wii = nn.Linear(self.d_model, self.d_model_)
        self.whi = nn.Linear(self.d_model_, self.d_model_)
        self.sigmi = nn.Sigmoid()

        # f
        self.wif = nn.Linear(self.d_model, self.d_model_)
        self.whf = nn.Linear(self.d_model_, self.d_model_)
        self.sigmf = nn.Sigmoid()

        # g
        self.wig = nn.Linear(self.d_model, self.d_model_)
        self.whg = nn.Linear(self.d_model_, self.d_model_)
        self.tanhg = nn.Tanh()

        # o
        self.wio = nn.Linear(self.d_model, self.d_model_)
        self.who = nn.Linear(self.d_model_, self.d_model_)
        self.sigmo = nn.Sigmoid()

        self.wq, self.wk, self.wv, self.wo = [
            nn.Linear(self.d_model_, self.d_model_) for _ in range(4)
        ]

        self.fc = nn.Sequential(
            nn.Linear(self.d_model_, 2 * self.d_model_),
            nn.ReLU(),
            nn.Linear(2 * self.d_model_, self.d_model_)
        )

        if self.use_gpu:
            self.gen = torch.randn(self.n_gts, self.d_model_).cuda()
        else:
            self.gen = torch.randn(self.n_gts, self.d_model_)

        #self.lstm = nn.LSTM(self.d_model, self.d_model_, 1)
        self.clear_states()

    def forward(self, x, y):
        """
        x: torch(n_frames, d_model)       |     frames
        y: torch(n_gts, bb_dim)           |     gts
        """

        # torch(d_model)
        now_frame = x[0]

        # torch(n_gts, d_model_)
        x = torch.cat([x[1:], y], dim=1)

        self.h = self.h.detach()

        # torch(d_model_)
        i = self.sigmi(self.wii(now_frame) + self.whi(self.h))
        f = self.sigmf(self.wif(now_frame) + self.whf(self.h))
        g = self.tanhg(self.wig(now_frame) + self.whg(self.h))
        o = self.sigmo(self.wio(now_frame) + self.who(self.h))

        """
        -----------------------------------------------
                            MH Attention
        -----------------------------------------------
        """

        # torch(hh, n_gts, d_model_ // h)
        queries, keys, values = [
            l(x).t().view(self.hh, self.d_model_ // self.hh, self.n_gts).transpose(1, 2)
            for l in (self.wq, self.wk, self.wv)
        ]

        # torch(hh, n_gts, n_frames)
        scores = F.softmax(torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.d_model_), dim=1)

        # torch(hh, n_gts, d_model_ // h)
        z = torch.matmul(scores, values)

        # torch(n_gts, d_model_)
        z = self.wo(z.transpose(0, 1).contiguous().view(self.n_gts, -1))
        z = self.fc(z)

        # torch(d_model_)
        c = (self.gen * z).sum(0)

        # torch(d_model_)
        self.h = o * self.tanh(f * c + i * g)

        return self.h[-self.bb_dim:]

    def clear_states(self):
        if self.use_gpu:
            self.h = torch.zeros(self.d_model_).cuda()
        else:
            self.h = torch.zeros(self.d_model_)


class MakeLSTM(nn.Module):

    def __init__(self, d_model, n_frames, n_gts, bb_dim):
        super(MakeLSTM, self).__init__()

        self.d_model = d_model
        self.n_frames = n_frames
        self.n_gts = n_gts
        self.bb_dim = bb_dim
        self.relu = nn.ReLU()
        self.use_gpu = torch.cuda.is_available()

        self.inner_dim = self.d_model + self.bb_dim

        self.clear_states()

        self.lstm = nn.LSTM(self.inner_dim, self.inner_dim, 1)

    def forward(self, x, y):
        """
        x: torch(n_frames, d_model)       |     frames
        y: torch(n_gts, bb_dim)           |     gts
        """

        # torch(1, 1, d_model)
        now_frame = x[0].unsqueeze(0).unsqueeze(0)
        # torch(1, 1, bb_dim)
        now_gt = y[0].unsqueeze(0).unsqueeze(0)
        # torch(1, 1, inner_dim)
        now_input = torch.cat([now_frame, now_gt], dim=2)

        x[1:] = x[1:].detach()
        y[1:] = y[1:].detach()

        """
        =============================
                forward pass
        =============================
        """

        self.h = self.h.detach()
        self.c = self.c.detach()

        out, (self.h, self.c) = self.lstm(now_frame, (self.h, self.c))

        return out.view(-1)[-self.bb_dim:]

    def clear_states(self):
        if self.use_gpu:
            self.h = torch.zeros(1, 1, self.inner_dim, requires_grad=False).cuda()
            self.c = torch.zeros(1, 1, self.inner_dim, requires_grad=False).cuda()
        else:
            self.h = torch.zeros(1, 1, self.inner_dim, requires_grad=False)
            self.c = torch.zeros(1, 1, self.inner_dim, requires_grad=False)


class MakeTransformerAdvanced(nn.Module):
    def __init__(self, d_model, n_frames, n_gts, bb_dim):
        super(MakeTransformerAdvanced, self).__init__()

        self.d_model = d_model
        self.n_frames = n_frames
        self.n_gts = n_gts
        self.bb_dim = bb_dim
        self.enc_ff = 2 * d_model
        self.dec_ff = 2 * bb_dim
        self.relu = nn.ReLU()

        # Decoder 1
        self.enc1_wq = nn.Linear(self.d_model, self.d_model)
        self.enc1_wk = nn.Linear(self.d_model, self.d_model)
        self.enc1_wv = nn.Linear(self.d_model, self.d_model)
        self.enc1_fc1 = nn.Linear(self.d_model, self.enc_ff)
        self.enc1_fc2 = nn.Linear(self.enc_ff, self.d_model)
        self.enc1_ln = nn.LayerNorm(self.d_model)
        self.enc1_dp = nn.Dropout(p=0.2)

        # Decoder 2
        self.enc2_wq = nn.Linear(self.d_model, self.d_model)
        self.enc2_wk = nn.Linear(self.d_model, self.d_model)
        self.enc2_wv = nn.Linear(self.d_model, self.d_model)
        self.enc2_fc1 = nn.Linear(self.d_model, self.enc_ff)
        self.enc2_fc2 = nn.Linear(self.enc_ff, self.d_model)
        self.enc2_ln = nn.LayerNorm(self.d_model)
        self.enc2_dp = nn.Dropout(p=0.2)

        # DecoderSAT 1
        self.decsat1_wq = nn.Linear(self.bb_dim, self.bb_dim)
        self.decsat1_wk = nn.Linear(self.bb_dim, self.bb_dim)
        self.decsat1_wv = nn.Linear(self.bb_dim, self.bb_dim)
        self.decsat1_wo = nn.Linear(self.bb_dim, self.bb_dim)

        self.aux = nn.Linear(self.n_frames, n_gts)

        # Decoder 1
        self.dec1_wq = nn.Linear(self.d_model, self.d_model)
        self.dec1_wk = nn.Linear(self.d_model, self.d_model)
        self.dec1_wv = nn.Linear(self.bb_dim, self.bb_dim)
        self.dec1_fc1 = nn.Linear(self.bb_dim, self.dec_ff)
        self.dec1_fc2 = nn.Linear(self.dec_ff, self.bb_dim)
        self.dec1_ln = nn.LayerNorm(self.bb_dim)
        self.dec1_dp = nn.Dropout(p=0.2)

        # DecoderSAT 2
        self.decsat2_wq = nn.Linear(self.bb_dim, self.bb_dim)
        self.decsat2_wk = nn.Linear(self.bb_dim, self.bb_dim)
        self.decsat2_wv = nn.Linear(self.bb_dim, self.bb_dim)
        self.decsat2_wo = nn.Linear(self.bb_dim, self.bb_dim)

        # Decoder 2
        self.dec2_wq = nn.Linear(self.d_model, self.d_model)
        self.dec2_wk = nn.Linear(self.d_model, self.d_model)
        self.dec2_wv = nn.Linear(self.bb_dim, self.bb_dim)
        self.dec2_fc1 = nn.Linear(self.bb_dim, self.dec_ff)
        self.dec2_fc2 = nn.Linear(self.dec_ff, self.bb_dim)
        self.dec2_ln = nn.LayerNorm(self.bb_dim)
        self.dec2_dp = nn.Dropout(p=0.2)

        # bb generator
        self.proj = nn.Linear(self.n_gts, self.n_gts)

    def forward(self, x, y):
        """
        x: torch(n_frames, d_model)       |     frames
        y: torch(n_gts, bb_dim)           |     gts
        """
        # стак 2 энкодеров и 2 декодеров ^
        # resiudal соедененид ^
        # layernorm
        # dropout(p=0.2)

        """
        -----------------------------------------------
                            ENCODER # 1
        -----------------------------------------------
        """
        # torch(n_frames, d_model)
        enc1_queries, enc1_keys, enc1_values = [
            l(x) for l in (self.enc1_wq, self.enc1_wk, self.enc1_wv)
        ]

        # torch(n_frames, n_frames)
        enc1_scores = F.softmax(torch.matmul(enc1_queries, enc1_keys.t()) / math.sqrt(self.d_model), dim=1)

        # torch(n_frames, d_model)
        enc1_z = torch.matmul(enc1_scores, enc1_values)

        # torch(n_frames, d_model)
        enc1_z = self.enc1_ln(enc1_z)
        enc1_z = x + self.enc1_fc2(self.enc1_dp(self.relu(self.enc1_fc1(enc1_z))))

        """
        -----------------------------------------------
                            ENCODER # 2
        -----------------------------------------------
        """
        # torch(n_frames, d_model)
        enc2_queries, enc2_keys, enc2_values = [
            l(enc1_z) for l in (self.enc2_wq, self.enc2_wk, self.enc2_wv)
        ]

        # torch(n_frames, n_frames)
        enc2_scores = F.softmax(torch.matmul(enc2_queries, enc2_keys.t()) / math.sqrt(self.d_model), dim=1)

        # torch(n_frames, d_model)
        enc2_z = torch.matmul(enc2_scores, enc2_values)

        # torch(n_frames, d_model)
        enc2_z = self.enc2_ln(enc2_z)
        enc2_z = enc1_z + self.enc2_fc2(self.enc2_dp(self.relu(self.enc2_fc1(enc2_z))))

        """
        -----------------------------------------------
                    DECODER SELF-ATTENTION # 1
        -----------------------------------------------
        """

        # torch(n_gts, bb_dim)
        decsat1_queries, decsat1_keys, decsat1_values = [
            l(y) for l in (self.decsat1_wq, self.decsat1_wk, self.decsat1_wv)
        ]

        # torch(n_gts, n_gts)
        decsat1_scores = F.softmax(torch.matmul(decsat1_queries, decsat1_keys.t()) / math.sqrt(self.bb_dim), dim=1)

        # torch(n_gts, bb_dim)
        decsat1_z = torch.matmul(decsat1_scores, decsat1_values)

        # torch(n_gts, bb_dim)
        decsat1_z = self.decsat1_wo(decsat1_z)

        """
        -----------------------------------------------
                            DECODER # 1
        -----------------------------------------------
        """

        # torch(n_gts, d_model)
        enc2_z = self.aux(enc2_z.t()).t()

        # torch(n_gts d_model)
        # torch(n_gts, d_model)
        # torch(n_gts, bb_dim)
        dec1_queries = self.dec1_wq(enc2_z)
        dec1_keys = self.dec1_wk(enc2_z)
        dec1_values = self.dec1_wv(decsat1_z)

        # torch(n_gts, n_gts)
        dec1_scores = F.softmax(torch.matmul(dec1_queries, dec1_keys.t()) / math.sqrt(self.d_model), dim=1)

        # torch(n_gts, bb_dim)
        dec1_z = torch.matmul(dec1_scores, dec1_values)

        # torch(n_gts, bb_dim)
        dec1_z = self.dec1_ln(dec1_z)
        dec1_z = self.dec1_fc2(self.dec1_dp(self.relu(self.dec1_fc1(dec1_z))))

        """
        -----------------------------------------------
                    DECODER SELF-ATTENTION # 2
        -----------------------------------------------
        """

        # torch(n_gts, bb_dim)
        decsat2_queries = self.decsat2_wq(dec1_z)
        decsat2_keys = self.decsat2_wk(dec1_z)
        decsat2_values = self.decsat2_wv(dec1_z)

        # torch(n_gts, n_gts)
        decsat2_scores = F.softmax(torch.matmul(decsat2_queries, decsat2_keys.t()) / math.sqrt(self.bb_dim), dim=1)

        # torch(n_gts, bb_dim)
        decsat2_z = torch.matmul(decsat2_scores, decsat2_values)

        # torch(n_gts, bb_dim)
        decsat2_z = self.decsat2_wo(decsat2_z)

        """
        -----------------------------------------------
                            DECODER # 2
        -----------------------------------------------
        """

        # torch(n_gts, d_model)
        # torch(n_gts, d_model)
        # torch(n_gts, bb_dim)
        dec2_queries = self.dec2_wq(enc2_z)
        dec2_keys = self.dec2_wk(enc2_z)
        dec2_values = self.dec2_wv(decsat2_z)

        # torch(n_gts, n_gts)
        dec2_scores = F.softmax(torch.matmul(dec2_queries, dec2_keys.t()) / math.sqrt(self.d_model), dim=1)

        # torch(n_gts, bb_dim)
        dec2_z = torch.matmul(dec2_scores, dec2_values)

        # torch(n_gts, bb_dim)
        dec2_z = self.dec2_ln(dec2_z)
        dec2_z = decsat2_z + self.dec2_fc2(self.dec2_dp(self.relu(self.dec2_fc1(dec2_z))))

        """
        -----------------------------------------------
                        BB GENERATOR
        -----------------------------------------------
        """

        return self.proj(dec2_z.t()).mean(1)


class MakeNet(nn.Module):

    def __init__(self, cnn, transformer, seq_len):
        super(MakeNet, self).__init__()
        self.cnn = cnn
        self.transformer = transformer
        self.seq_len = seq_len
        self.is_init = True
        self.frames, self.gts = None, None
        self.aux_clear = getattr(self.transformer, "clear_states", None)

    def forward(self):
        """
        transformer forward pass
        :return: torch(5)
        """
        return self.transformer(self.frames, self.gts)

    def refresh(self, frame, gt):
        """
        refresh historical data after load new frame and gt
        :param frame: torch(3, width, height)
        :param gt: torch(5)
        """
        if self.is_init:
            self.frames = self.init_seq(self.cnn(frame.unsqueeze(0)), self.seq_len + 1)
            self.gts = self.init_seq(gt.unsqueeze(0), self.seq_len).detach()
            self.is_init = False
        else:
            self.gts = self.init_seq(torch.zeros(1, 5), self.seq_len).detach()
            self.frames = self.pull_frames(self.cnn(frame.unsqueeze(0)))

    def clear(self):
        self.is_init = True
        self.gts = None
        self.frames = None
        if callable(self.aux_clear):
            self.aux_clear()

    @staticmethod
    def init_seq(elem, l):
        """
        create initial seq for transformer
        :param elem: torch(1, some_size)
        :param len: expanding size, int
        """
        return elem.expand(l, elem.size(1))

    def pull_frames(self, new_frame):
        """
        cycle permute along dim=0 + new frame tensor
        :param new_frame: torch(1, )
        """
        return torch.cat([new_frame, self.frames[:-1]], dim=0).detach()

    def pull_gts(self, gt):
        """
        cycle permute along dim=0 + new gt tensor
        :param gt: torch(1, 5)
        """
        self.gts = torch.cat([gt, self.gts[:-1]], dim=0).detach()


if __name__ == '__main__':
    """some tests"""
    n_gts = 5
    n_frames = n_gts + 1
    d_model = 11
    h = 2
    bb_dim = 5

    frames = torch.randn(n_frames, d_model)
    gts = torch.randn(n_gts, bb_dim)

    net = MakeTransformer(d_model, n_frames, n_gts, bb_dim)

    ans = net(frames, gts)
    print(ans.size())

