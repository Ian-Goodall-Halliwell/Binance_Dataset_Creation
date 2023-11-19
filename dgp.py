import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from workingdir import WORKING_DIR
import os
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
batch_size = 1024
smoke_test = False
i = 5
with open("data_new.pkl","rb") as f:
        train_x,train_y,test_x,test_y = pkl.load(f)
from sklearn.preprocessing import QuantileTransformer
train_x = train_x.xs(train_x.axes[0].levels[1][-1],level=1)
test_x = test_x.xs(test_x.axes[0].levels[1][-1],level=1)
train_y = train_y.xs(train_y.axes[0].levels[1][-1],level=1)
test_y = test_y.xs(test_y.axes[0].levels[1][-1],level=1)
train_x = train_x.dropna(how="all")
test_x = test_x.dropna(how="all")
train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

train_x = train_x.fillna(method='ffill')
test_x = test_x.fillna(method='ffill')
train_x = train_x.fillna(0)
test_x = test_x.fillna(0)
train_x = train_x.dropna(how="any")
test_x = test_x.dropna(how="any")
train_y = train_y.reindex(train_x.index)
test_y = test_y.reindex(test_x.index)
train_y = train_y.sort_index()
test_x = test_x.sort_index()
test_y = test_y.sort_index()
train_x = train_x.sort_index()
# tmodel = QuantileTransformer(n_quantiles=10000, output_distribution="normal").fit(train_x)
# X_train = tmodel.transform(train_x)
# X_test = tmodel.transform(test_x)
#standard = StandardScaler()
scale = MinMaxScaler(feature_range=(-1,1)).fit(train_x)
train_x = scale.transform(train_x).squeeze()
test_x = scale.transform(test_x).squeeze()
# scale = MinMaxScaler(feature_range=(-1,1)).fit(train_y[:,i].reshape(-1, 1))
# train_y = scale.transform(train_y[:,i].reshape(-1, 1)).squeeze()
# test_y = scale.transform(test_y[:,i].reshape(-1, 1)).squeeze()

try:
    train_y = train_y[:,i]
    test_y = test_y[:,i]
except:
    pass
print(test_y.std())
print(test_y.mean())
# print(scale.transform(np.array([0]).reshape(-1, 1)))
try:
    train_x,train_y,test_x,test_y = torch.Tensor(train_x),torch.Tensor(train_y),torch.Tensor(test_x),torch.Tensor(test_y)
except:
    train_x,train_y,test_x,test_y = torch.Tensor(train_x),torch.Tensor(train_y.values.squeeze()),torch.Tensor(test_x),torch.Tensor(test_y.values.squeeze())

# if torch.cuda.is_available():
#     train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
num_tasks = train_y.size(-1)
train_n = len(train_x)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=512, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))
num_hidden_dims = 2 if smoke_test else 2


class DeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            confslow = []
            confshigh = []
            for x_batch, y_batch in test_loader:
                x_batch,y_batch = x_batch.cuda(),y_batch.cuda()
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))
                conpred = preds.confidence_region()
                confslow.append(conpred[0].cpu())
                confshigh.append(conpred[1].cpu())

        return torch.cat(mus, dim=-1).mean(0).cpu().detach().numpy(), torch.cat(variances, dim=-1).mean(0).cpu().detach().numpy(), torch.cat(lls, dim=-1).mean(0).cpu().mean().detach().numpy(), torch.cat(confslow, dim=-1).mean(0).cpu().detach().numpy(), torch.cat(confshigh, dim=-1).mean(0).cpu().detach().numpy()
    
    
    
model = DeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()
num_epochs = 1 if smoke_test else 10
num_samples = 3 if smoke_test else 50


optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        x_batch,y_batch = x_batch.cuda(),y_batch.cuda()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())
import gpytorch
import math


test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

model.eval()
predictive_means, predictive_variances, test_lls, confls, confhs = model.predict(test_loader)
#confls = scale.inverse_transform(confls.reshape(-1, 1))
#confhs = scale.inverse_transform(confhs.reshape(-1, 1))
from sklearn.metrics import mean_absolute_error, mean_squared_error
test_y = test_y.detach().numpy()
rmse = math.sqrt(mean_squared_error(test_y,predictive_means))
#rmse = torch.mean(torch.pow(predictive_means.cpu().mean(0) - test_y, 2)).sqrt()
mae = mean_absolute_error(test_y,predictive_means)
from sklearn.metrics import r2_score
score = r2_score(test_y, predictive_means)
print(score)
#mae = torch.mean(predictive_means.cpu().mean(0) - test_y)
#mns = predictive_means.cpu().mean(0).detach().numpy()
#test_y = test_y.detach().numpy()
# from sklearn.metrics import mean_absolute_error
# mae = mean_absolute_error(mns,test_y)
torch.save(model.state_dict(),"modelparams.torch")
print(f"RMSE: {rmse}, MAEE: {mae}, NLL: {-test_lls}")