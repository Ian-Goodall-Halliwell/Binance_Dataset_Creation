import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood,MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, BatchDecoupledVariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
import gpytorch.settings as settings
import os
from workingdir import WORKING_DIR
import pickle as pkl
from math import floor
from sklearn.preprocessing import MinMaxScaler
# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)

batch_size = 2000                 # Size of minibatch
milestones = [10, 50, 90]       # Epochs at which we will lower the learning rate by a factor of 0.1
num_inducing_pts = 300            # Number of inducing points in each hidden layer
num_epochs = 10                   # Number of epochs to train for
initial_lr = 0.01                 # Initial learning rate
hidden_dim = 2                # Number of GPs (i.e., the width) in the hidden layer.
num_quadrature_sites = 8          # Number of quadrature sites (see paper for a description of this. 5-10 generally works well).

## Modified settings for smoke test purposes
num_epochs = 1 if smoke_test else num_epochs

with open(os.path.join(WORKING_DIR, "full_data/dataset_.pkl"),"rb") as f:
                train_x,train_y,test_x,test_y = pkl.load(f)
i = 6
scale = MinMaxScaler(feature_range=(-1,1)).fit(train_x)
train_x = scale.transform(train_x)
test_x = scale.transform(test_x)
train_y *= 100
test_y *= 100
test_y = test_y[:,i]
train_y = train_y[:,i]
# scale = MinMaxScaler(feature_range=(-1,1)).fit(train_y[:,i].reshape(-1, 1))
# train_y = scale.transform(train_y[:,i].reshape(-1, 1)).squeeze()
# test_y = scale.transform(test_y[:,i].reshape(-1, 1)).squeeze()
print(test_y.std())
print(test_y.mean())
train_x,train_y,test_x,test_y = torch.Tensor(train_x),torch.Tensor(train_y),torch.Tensor(test_x),torch.Tensor(test_y)
# if torch.cuda.is_available():
#     train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
num_tasks = train_y.size(-1)
train_n = len(train_x)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from scipy.cluster.vq import kmeans2

# Use k-means to initialize inducing points (only helpful for the first layer)
inducing_points = (train_x[torch.randperm(min(1000 * 100, train_n))[0:num_inducing_pts], :])
inducing_points = inducing_points.clone().data.cpu().numpy()
inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),
                               inducing_points, minit='matrix')[0])

if torch.cuda.is_available():
    inducing_points = inducing_points.cuda()
    
class DSPPHiddenLayer(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TwoLayerDSPP(DSPP):
    def __init__(self, train_x_shape, inducing_points, num_inducing, hidden_dim=3, Q=3):
        hidden_layer = DSPPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            mean_type='linear',
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )

        likelihood = GaussianLikelihood(num_tasks=num_tasks)

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs, **kwargs)
        output = self.last_layer(hidden_rep1, **kwargs)
        return output

    def predict(self, loader):
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                x_batch,y_batch = x_batch.cuda(),y_batch.cuda()
                preds = self.likelihood(self(x_batch, mean_input=x_batch))#.to_data_independent_dist()
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                # Compute test log probability. The output of a DSPP is a weighted mixture of Q Gaussians,
                # with the Q weights specified by self.quad_weight_grid. The below code computes the log probability of each
                # test point under this mixture.

                # Step 1: Get log marginal for each Gaussian in the output mixture.
                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))

                # Step 2: Weight each log marginal by its quadrature weight in log space.
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll

                # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

model = TwoLayerDSPP(
    train_x.shape,
    inducing_points,
    num_inducing=num_inducing_pts,
    hidden_dim=hidden_dim,
    Q=num_quadrature_sites
)

if torch.cuda.is_available():
    model.cuda()

model.train()



# We use the adam optimizer with a learning rate of 0.01
from gpytorch.mlls import DeepPredictiveLogLikelihood

adam = torch.optim.Adam([{'params': model.parameters()}], lr=initial_lr, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=milestones, gamma=0.1)

# The "beta" parameter here corresponds to \beta_{reg} from the paper, and represents a scaling factor on the KL divergence
# portion of the loss.
objective = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=train_n, beta=0.05)

import tqdm

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

for i in epochs_iter:
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        x_batch,y_batch = x_batch.cuda(),y_batch.cuda()
        adam.zero_grad()
        output = model(x_batch)
        loss = -objective(output, y_batch)
        loss.backward()
        adam.step()
    
    sched.step()
del x_batch,y_batch
    
model.eval()
means, vars, ll = model.predict(test_loader)
weights = model.quad_weights.unsqueeze(-1).exp().cpu()
mns = (weights * means).sum(0).cpu().detach().numpy()
test_y = test_y.cpu().detach().numpy()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
rmse = math.sqrt(mean_squared_error(test_y,mns))
#rmse = torch.mean(torch.pow(predictive_means.cpu().mean(0) - test_y, 2)).sqrt()
mae = mean_absolute_error(test_y,mns)
from sklearn.metrics import r2_score
score = r2_score(test_y,mns)
print(score)
# `means` currently contains the predictive output from each Gaussian in the mixture.
# To get the total mean output, we take a weighted sum of these means over the quadrature weights.
# rmse = ((weights * means).sum(0) - test_y.cpu()).pow(2.0).mean().sqrt().item()
# mae = ((weights * means).sum(0) - test_y.cpu()).mean(0)
ll = ll.mean().item()
torch.save(model.state_dict(),"modelparams.torch")
print('RMSE: ', rmse,'MAE: ', mae, 'Test NLL: ', -ll)