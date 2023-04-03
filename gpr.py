import gpytorch
import torch
from tqdm import tqdm
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self,train_loader,likelihood,num_data,num_epochs):
        


        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, self, num_data=num_data)


        epochs_iter = tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                optimizer.zero_grad()
                output = self(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
                
    def predict(self,test_loader,likelihood):
        self.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        confshigh = torch.tensor([0.])
        confslow = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                preds = self(x_batch)
                conpred = preds.confidence_region()
                confslow = torch.cat([confslow, conpred[0].cpu()])
                confshigh = torch.cat([confshigh, conpred[1].cpu()])
                means = torch.cat([means, preds.mean.cpu()])
        means = means[1:]
        return means, confslow,confshigh
        print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))