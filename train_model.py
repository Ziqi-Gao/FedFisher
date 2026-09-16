from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag


class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args["bs"], shuffle=True)

    def train_and_compute_fisher(self, net, n_c, timing=None, *,
                                 fisher_kind="both", isolate_fisher_rng=False):
        # Timing is observational and opt-in. The caller owns the outer train
        # stage. Defaults retain both Fishers in their original order; the
        # standalone CPU recipe requests one representation and isolates its
        # shuffled-loader RNG consumption from subsequent local SGD fits.
        if fisher_kind not in ("both", "diag", "kfac"):
            raise ValueError("fisher_kind must be both, diag or kfac")
        if type(isolate_fisher_rng) is not bool:
            raise ValueError("isolate_fisher_rng must be boolean")
        if isolate_fisher_rng and str(self.args.get("device", "cuda")) != "cpu":
            raise ValueError("isolated Fisher RNG is supported only by the CPU recipe")
        with timing.measure("train", component="sgd") if timing is not None else nullcontext():
            net.train()
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args["eta"], momentum=0.9)

            for epoch in range(self.args["local_epochs"]):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args["device"]), labels.to(self.args["device"])
                    optimizer.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                print("Epoch No. ", epoch, "Loss ", sum(batch_loss) / len(batch_loss))

        F_kfac, F_diag = None, None
        with torch.random.fork_rng(devices=[]) if isolate_fisher_rng else nullcontext():
            if fisher_kind in ("both", "kfac"):
                with timing.measure("train", component="fisher_kfac") if timing is not None else nullcontext():
                    F_kfac = FIM(
                        model=net,
                        loader=self.ldr_train,
                        representation=PMatKFAC,
                        device=self.args.get("device", "cuda"),
                        n_output=n_c,
                    )

            if fisher_kind in ("both", "diag"):
                with timing.measure("train", component="fisher_diag") if timing is not None else nullcontext():
                    F_diag = FIM(
                        model=net,
                        loader=self.ldr_train,
                        representation=PMatDiag,
                        device=self.args.get("device", "cuda"),
                        n_output=n_c,
                    )

                    F_diag = F_diag.get_diag()
        vec_curr = parameters_to_vector(net.parameters())
        return vec_curr, net, F_kfac, F_diag
