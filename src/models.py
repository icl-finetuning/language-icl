import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import os

import numpy as np

from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf, seq):
    if conf.family == "gpt2":
        if "garg" not in conf.name:
            if not conf.language_finetune:
                if not conf.preconfig:
                    print("Building a model from pre-trained GPT2 language model")
                    model = FromLanguageTransformerModel(conf.n_dims, family=conf.family, checkpoint=conf.name, n_embd=conf.n_embd, mlp=conf.mlp, freeze_ln=conf.freeze_ln, pca=conf.pca, seq=seq)
                else:
                    print("Building a synthetic model from scratch")
                    model = TransformerModel(conf.n_dims, conf.n_positions, preconfigured=conf.name, n_embd=conf.n_embd, freeze_ln=conf.freeze_ln, seq=seq)
            else:
                print("Building language model from a pre-trained GPT2 synthetic model")  
                model = FromSyntheticTransformerModel(conf.n_dims, conf.synth_ckpt, conf.name, conf.n_positions, family=conf.family,  n_embd=conf.n_embd, freeze_ln=conf.freeze_ln, seq=seq, random_init=conf.random_init)
        else:
            model = TransformerModel(
                n_dims=conf.n_dims,
                n_positions=conf.n_positions,
                n_embd=conf.n_embd,
                n_layer=conf.n_layer,
                n_head=conf.n_head,
                seq=seq
            )
    else:
        raise NotImplementedError(f"Model not implemented.")

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        "seq_linear": [],
        "seq_relu_2nn": [], 
        "seq_rec_linear": []
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

SEQ = True
class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, freeze_ln=False, seq=False, preconfigured=""):
        super(TransformerModel, self).__init__()
        if preconfigured != "":
            configuration = GPT2Config.from_pretrained(preconfigured)
        else:
            configuration = GPT2Config(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self.seq = seq
        if self.seq:
            self._read_out = nn.Linear(n_embd, n_dims) 
        else: 
            self._read_out = nn.Linear(n_embd, 1)
        
        if freeze_ln:
            # freeze all the parameters of the GPT2 backbone 
            for name, param in self._backbone.named_parameters():
                if 'wte' not in name and 'wpe' not in name:
                    param.requires_grad = False
        else: 
            for name, param in self._backbone.named_parameters():
                if 'wte' not in name and 'wpe' not in name and 'ln' not in name:
                    param.requires_grad = False

    @staticmethod
    def _combine(xs_b, ys_b, seq):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        # print("is seq", seq)
        if not seq:
            ys_b = torch.cat(
                (
                    ys_b.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)

        return zs

        # zs = torch.empty(xs_b.size(0), xs_b.size(1) + ys_b.size(1), xs_b.size(2), dtype=xs_b.dtype, device=xs_b.device)
        # zs[:, ::2, :] = xs_b
        # zs[:, 1::2, :] = ys_b
        # return zs
        

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys, self.seq)
        # print(zs)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        if self.seq:
            return prediction[:, ::2, :][:, inds]  # predict only on xs
        else:
            return prediction[:, ::2, 0][:, inds]

class FromLanguageTransformerModel(nn.Module):
    def __init__(self, n_dims, family="gpt2", checkpoint="openai-community/gpt2", n_embd=128, mlp=False, freeze_ln=False, pca=False, seq=False):
        super(FromLanguageTransformerModel, self).__init__()

        # there is no need for a GPT2 configuration if you use a pretrained model.
        self.name = family

        self.n_dims = n_dims

        if mlp:
            self._read_in = nn.Sequential(
                nn.Linear(n_dims, n_dims*2),
                nn.ReLU(),
                nn.Linear(n_dims * 2, n_dims*4),
                nn.ReLU(),
                nn.Linear(n_dims * 4, n_embd)
            )
        else:
            self._read_in = nn.Linear(n_dims, n_embd)

        self.pca = pca
        if self.pca:
            print("using PCA.")
            if n_embd == 768:
                self.pca_projection = nn.Parameter(torch.from_numpy(np.load("../pca_small.npy")), requires_grad=False)
            elif n_embd == 1024:
                self.pca_projection = nn.Parameter(torch.from_numpy(np.load("../pca_med.npy")), requires_grad=False)
            else:
                raise NotImplementedError("PCA dimension not implemented!")
            print(self.pca_projection.shape)
        if family == "gpt2":
            self._backbone = GPT2Model.from_pretrained(checkpoint)
        # elif family == "Llama-2":
        #     self._backbone = LlamaForCausalLM.from_pretrained(checkpoint)

        print(self._backbone.config, "model config")
        if freeze_ln:
            # freeze all the parameters of the GPT2 backbone 
            for name, param in self._backbone.named_parameters():
                if 'wte' not in name and 'wpe' not in name:
                    param.requires_grad = False
        else: 
            for name, param in self._backbone.named_parameters():
                if 'wte' not in name and 'wpe' not in name and 'ln' not in name:
                    param.requires_grad = False

        self.seq = seq
        if self.seq:
            self._read_out = nn.Linear(n_embd, n_dims) 
        else: 
            self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b, seq):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        if not seq:
            ys_b = torch.cat(
                (
                    ys_b.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=ys_b.device),
                ),
                axis=2,
            )
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)

        return zs

        # zs = torch.empty(xs_b.size(0), xs_b.size(1) + ys_b.size(1), xs_b.size(2), dtype=xs_b.dtype, device=xs_b.device)
        # zs[:, ::2, :] = xs_b
        # zs[:, 1::2, :] = ys_b
        # return zs
        

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys, self.seq)
        # print(zs)
        embeds = self._read_in(zs)
        ## do pca
        if self.pca:
            embeds = embeds @ self.pca_projection.T
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        if self.seq:
            return prediction[:, ::2, :][:, inds]  # predict only on xs
        else:
            return prediction[:, ::2, 0][:, inds]
    
class FromSyntheticTransformerModel(nn.Module):
    def __init__(self, n_dims, synth_ckpt_dir, pretrained_language_ckpt, n_positions, n_embd=128, n_layer=12, n_head=4, family="gpt2", freeze_ln=False, seq=False, random_init=False):
        super(FromSyntheticTransformerModel, self).__init__()

        self.name = family

        self.n_dims = n_dims
        
        # finetune just the layer norms and the output unembedding? maybe? not sure how large that is
        synthetic_model = TransformerModel(n_dims, n_positions, n_embd, n_layer, n_head, seq)
       
        # load synthetic model
        if not random_init:
            state_path = os.path.join(synth_ckpt_dir, "state.pt")
            print("where the synthetic_model is", state_path)
            if os.path.exists(state_path):
                state = torch.load(state_path)
                synthetic_model.load_state_dict(state["model_state_dict"])
            else: 
                # TODO: change this error
                raise ValueError('Model path does not exist')

        self._backbone = synthetic_model._backbone

        pretrained_lang_model = GPT2Model.from_pretrained(pretrained_language_ckpt)

        self._backbone.wte = pretrained_lang_model.wte

        self.classifier = torch.nn.Linear(n_embd, 2)


        if freeze_ln:
            # freeze all the parameters of the GPT2 backbone 
            for name, param in self._backbone.named_parameters():
                # TODO: do we also unfreeze 
                if 'wpe' not in name:
                    param.requires_grad = False
        else: 
            for name, param in self._backbone.named_parameters():
                if 'layer_norm' not in name and 'wpe' not in name:
                    param.requires_grad = False
    def forward(self, **args):
        args_without_labels = args.copy()
        labels = args_without_labels.pop('labels', None)

        outputs = self._backbone(**args_without_labels)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Get the pooled output
        logits = self.classifier(last_hidden_state)

        labels = args.get('labels')

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss, logits
        else:
            return logits



class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
