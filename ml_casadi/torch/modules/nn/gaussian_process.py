import gpytorch
import torch
import casadi as cs
from ml_casadi.torch.modules import TorchMLCasadiModule


class BatchIndependentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_size):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([output_size])),
            batch_shape=torch.Size([output_size])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.likelihood(gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        ))


class GPModel(TorchMLCasadiModule):
    def __init__(self, train_x, train_y, input_size, output_size, device="cuda:0", train_lr=1e-1, train_epochs=50):
        super().__init__()
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        self.input_size = input_size
        self.output_size = output_size
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_size).to(device)
        self.model = BatchIndependentGPModel(train_x, train_y, self.likelihood, output_size).to(device)

        self.likelihood.train()
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(train_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            print("GP epoch %d, loss: %.8f" % (i, loss))

        self.likelihood.eval()
        self.model.eval()

    # Ugly hacks to unify the api between NN/GP:
    def approx(self, x: cs.MX, order=1, gp_model=False, weakening_cutoff=0, weakening_strength=1, difference_weights=None):
        return super().approx(x, order, gp_model=True, weakening_cutoff=weakening_cutoff, weakening_strength=weakening_strength, difference_weights=difference_weights)
    def sym_approx_params(self, flat=False, order=1, gp_model=None):
        return super().sym_approx_params(flat, order, gp_model=True)
    def approx_params(self, a, flat=False, order=1, gp_model=None):
        return super().approx_params(a, flat, order, gp_model=True)

    def forward(self, x):
        with gpytorch.settings.fast_pred_var():
            out = self.likelihood(self.model(x))
            return torch.stack([out.mean, out.stddev], dim=1)