from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
import torch.distributions as dist
from pyro.distributions import Delta
import torch
import copy
from typing import Callable, Optional, Iterable
import cloudpickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__)

from recourse.distutils import DistAbdUnobs
    
class Seq:
    """Class implementing structural equations including the distribution of the noise term"""
    
    def __init__(self, node: str, parents: list[str], 
                 f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                 f_inv: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                 d: dist.Distribution,
                 description: Optional[str] = None):
        """
        parents: list of strings representing the ordered parents of the node
        f: function of the form (parents, eps) -> float
        f_inv: function of the form (parents, x) -> float
        """
        self.node = node
        self.parents = parents
        self.f = f
        self.f_inv = f_inv
        self.d = d
        self.description = description
        self.node2idx: Optional[dict[str, int]] = None
    
    @classmethod    
    def seqs_to_dict(cls, seqs: Iterable[Seq]) -> dict[str, Seq]:
        return {seq.node: seq for seq in seqs}
    
    @classmethod
    def seqs_to_edges(cls, seqs: Iterable[Seq]) -> list[tuple[str, str]]:
        edges = []
        for seq in seqs:
            for parent in seq.parents:
                edges.append((parent, seq.node))
        return edges
    
    def set_node2idx(self, node2idx: dict[str, int]):
        """
        Set the node2idx mapping for the sequence.
        This is used to map node names to their corresponding indices.
        """
        self.node2idx = node2idx
        
    def __call__(self, obss: torch.Tensor, epss_node: torch.Tensor) -> torch.Tensor:
        assert self.node2idx is not None
        pars = [self.node2idx[p] for p in self.parents]
        res = self.f(obss[:, pars], epss_node)
        return res
    
    def inv(self, obss: torch.Tensor) -> torch.Tensor:
        assert self.node2idx is not None
        if self.f_inv is None:
            raise ValueError("f_inv is not defined for this structural equation.")
        pars, ix = [self.node2idx[p] for p in self.parents], self.node2idx[self.node]
        res = self.f_inv(obss[:, pars], obss[:, ix])
        return res
        
    def sample_eps(self, sample_shape:torch.Size | tuple=torch.Size([]),
                   obss: Optional[torch.Tensor]=None,
                   epss: Optional[torch.Tensor]=None) -> torch.Tensor:
        if type(sample_shape) is tuple:
            sample_shape = torch.Size(sample_shape)
        assert type(sample_shape) is torch.Size
        epss_node = self.d.sample(sample_shape=sample_shape).float()
        return epss_node
    
    def sample(self, obss: torch.Tensor, 
               sample_shape: torch.Size | tuple=torch.Size([]),
               epss: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
        if type(sample_shape) is tuple:
            sample_shape = torch.Size(sample_shape)
        assert type(sample_shape) is torch.Size
        epss_node = self.sample_eps(sample_shape=sample_shape, obss=obss, epss=epss)
        obss_node = self(obss, epss_node)
        return obss_node, epss_node
    
    def log_prob_eps(self, epss_node: torch.Tensor, obss: Optional[torch.Tensor]=None) -> torch.Tensor:
        log_probs = self.d.log_prob(epss_node)
        return log_probs
    
    def log_prob(self, obss: torch.Tensor) -> torch.Tensor:
        assert self.node2idx is not None
        epss_node = self.inv(obss)
        log_probs = self.log_prob_eps(epss_node, obss=obss)
        return log_probs
    
    
class DependentNoiseSeq:
    """Class implementing SEQs where the noise is a function of other nodes
    
    For SCMs with invertible SEQs where one node was unobserved, the abducted noise
    for the children of the unobserved node is a function of the noise of the 
    unobserved node. This class implements this functionality, overriding the default
    behavior of the SEQ.
    """
    def __init__(self, seq: Seq, g_fn: Callable[[torch.Tensor], torch.Tensor], seq_dependent_var: Seq):
        super().__setattr__('_seq', seq)
        super().__setattr__('g_fn', g_fn)
        super().__setattr__('seq_dependent_var', seq_dependent_var)
        
    def sample_eps(self, sample_shape: torch.Size=torch.Size([]),
                   obss: Optional[torch.Tensor]=None,
                   epss: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Instead of sampling from a distribution, we simply compute it from 
        the eps values of the unobserved node.
        """
        assert epss is not None
        node_ix = self.seq_dependent_var.node2idx[self.seq_dependent_var.node]
        epss_node = self.g_fn(epss[:, node_ix])
        return epss_node
    
    def log_prob_eps(self, epss_node: torch.Tensor,
                     obss: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Instead of computing the logprob of a distribution,
        we simply check whether the eps_node is equal to the
        value that we would expect. We have some small error tolerance.
        """
        assert obss is not None
        # reconstruct noise for previously unobserved node
        eps_dependent_var = self.seq_dependent_var.inv(obss)
        correct_eps = self.g_fn(eps_dependent_var)
        same = torch.isclose(epss_node, correct_eps, atol=1e-5)
        log_probs = torch.log(same.float())
        return log_probs
    
    ## the following two methods are simply copied from the original Seq class
    ## this ensure that the methods call the DependentNoiseSeq methods instead of the Seq methods
    
    def sample(self, *args, **kwargs):
        return Seq.sample(self, *args, **kwargs) # type: ignore
    
    def log_prob(self, *args, **kwargs):
        return Seq.log_prob(self, *args, **kwargs) # type: ignore
    
    def __call__(self, *args, **kwds):
        return self._seq(*args, **kwds)
    
    # for all other methods its fine if the Seq version is used
        
    def __getattr__(self, attr):
        return getattr(self._seq, attr)

    def __setattr__(self, name, value):
        # If the attribute is one of our own, set it normally
        if name in {'_seq', 'g_fn', 'seq_dependent_var'}:
            super().__setattr__(name, value)
        else:
            # Otherwise, forward it to the wrapped Seq instance
            setattr(self._seq, name, value)
    
    def __deepcopy__(self, memo):
        # Create a deepcopy of the underlying _seq and other attributes explicitly.
        new_seq = copy.deepcopy(self._seq, memo)
        new_g_fn = copy.deepcopy(self.g_fn, memo)
        new_seq_dependent_var = copy.deepcopy(self.seq_dependent_var, memo)
        return type(self)(new_seq, new_g_fn, new_seq_dependent_var)

class LinGauSeq(Seq):
    """Class implementing a linear structural equation with Gaussian noise"""
    def __init__(self, node: str, parents: list[str],
                 f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 f_inv: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 d: dist.Distribution):
        """
        data: pd.DataFrame with columns node, parents, eps
        """
        super().__init__(node, parents, f, f_inv, d)
        self.sklearn_model = None
    
    @classmethod
    def fit(cls, data: pd.DataFrame, node: str, parents: list[str]) -> LinGauSeq:
        X = data[parents]
        y = data[node]
        y_torch = torch.tensor(y.to_numpy()).float()
        model = None
        if len(parents) == 0:
            predict_torch = lambda X: y_torch.mean()
            predict_local = lambda X: y.mean()
        else:
            model = LinearRegression().fit(X, y)
            w = torch.from_numpy(model.coef_).float()
            b = torch.tensor(model.intercept_).float()
            def predict_torch(X):
                assert X.dtype == torch.float
                return X @ w + b
            predict_local = lambda X: model.predict(X)
        resid = y - predict_local(X)
        d = dist.Normal(resid.mean(), resid.std())
        f = lambda par, eps: predict_torch(par) + eps
        f_inv = lambda par, x: x - predict_torch(par)
        
        seq = LinGauSeq(node, parents, f, f_inv, d)
        seq.sklearn_model = model
        return seq
    
    def plot_fit_diagnostic(self, data: torch.Tensor, savepath: str):
        ## compute residuals
        assert self.node2idx is not None
        zero_eps = torch.zeros(data.shape[0])
        node_ix = self.node2idx[self.node]
        pred = self(data, zero_eps)
        resid = data[:, node_ix] - pred
        
        ## plot linear fit in scatterplots
        plt.figure()
        sns.scatterplot(x=data[:, node_ix].numpy(), y=pred.numpy())
        plt.title('Predicted vs. Observed')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.savefig(savepath + f'lingam_fit_{self.node}.pdf')
        plt.close()
        
        ## plot the residuals vs the fitted noise
        epss_node = self.sample_eps(sample_shape=(resid.shape[0],))
        df = pd.DataFrame({'eps': resid})
        df['type'] = 'empirical'
        df2 = pd.DataFrame({'eps': epss_node.numpy()})
        df2['type'] = 'fit'
        df = pd.concat([df, df2])
        sns.displot(df, x='eps', hue='type', kde=True)
        plt.title('Residuals')
        plt.savefig(savepath + f'lingam_residuals_{self.node}.pdf')
        plt.close()
        
        sns.displot(df, x='eps', hue='type', kind='ecdf')
        plt.title('Residuals')
        plt.savefig(savepath + f'lingam_residuals_ecdf_{self.node}.pdf')
        plt.close()
        
        
        par_ixs = [self.node2idx[p] for p in self.parents]
        
        ## if exactly one parent, plot x vs y scatter and add a line for the linear fit
        if len(self.parents) == 1:
            plt.figure()
            sns.scatterplot(x=data[:, par_ixs[0]].numpy(), y=data[:, node_ix].numpy())
            sns.lineplot(x=data[:, par_ixs[0]].numpy(), y=pred.numpy(), color='red')
            plt.title('Predicted vs. Observed')
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.savefig(savepath + f'lingam_fit_{self.node}_scatter.pdf')
            plt.close()


class RFGauSeq(Seq):
    """Class implementing a linear structural equation with Gaussian noise"""
    def __init__(self, node: str, parents: list[str],
                 f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 f_inv: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 d: dist.Distribution):
        """
        data: pd.DataFrame with columns node, parents, eps
        """
        super().__init__(node, parents, f, f_inv, d)
        self.sklearn_model = None
    
    @classmethod
    def fit(cls, data: pd.DataFrame, node: str, parents: list[str]) -> RFGauSeq:
        X = data[parents]
        y = data[node]
        y_torch = torch.tensor(y.to_numpy()).float()
        model = None
        if len(parents) == 0:
            predict_torch = lambda X: y_torch.mean()
            predict_local = lambda X: y.mean()
        else:
            model = RandomForestRegressor(n_estimators=10, max_depth=10).fit(X, y)
            def predict_torch(X):
                # this does not really make sense here but we simply do it for consistency
                # makes more sense for linear model where the code comes from
                assert X.dtype == torch.float
                X_np = X.numpy()
                X_df = pd.DataFrame(X_np, columns=parents)
                pred = model.predict(X_df)
                pred = torch.from_numpy(pred).float()
                assert pred.dtype == torch.float
                return pred
            predict_local = lambda X: model.predict(X)
        resid = y - predict_local(X)
        d = dist.Normal(resid.mean(), resid.std())
        f = lambda par, eps: predict_torch(par) + eps
        f_inv = lambda par, x: x - predict_torch(par)
        
        seq = RFGauSeq(node, parents, f, f_inv, d)
        seq.sklearn_model = model
        return seq
    

class SCM:
    """Class implementing a structural causal model as a combination of Seqs"""
    def __init__(self, seqs):
        edges = Seq.seqs_to_edges(seqs)
        self.graph = nx.DiGraph(edges)
        self.seqs = Seq.seqs_to_dict(seqs)
        self.nodes = self.get_nodes_ordered()
        self.node2idx = {node: i for i, node in enumerate(self.get_nodes_ordered())}
        for seq in self.seqs.values():
            seq.set_node2idx(self.node2idx)
        self.obss: Optional[torch.Tensor] = None
        self.epss: Optional[torch.Tensor] = None
        self.abducted = False
        self.bounds = {}
        self.support = {}
        self._epss_is_sampled = False
        self._obss_is_sampled = False
    
    @classmethod    
    def save(cls, scm: SCM, path: str):
        """
        Save the SCM to a file.
        """
        with open(path, 'wb') as f:
            cloudpickle.dump(scm, f)
        logger.info(f"SCM saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> SCM:
        """
        Load the SCM from a file.
        """
        with open(path, 'rb') as f:
            scm = cloudpickle.load(f)
        logger.info(f"SCM loaded from {path}")
        return scm
        
    def clear_sample(self, sample_shape=()):
        if len(sample_shape) == 0:
            self.obss = torch.zeros((1, len(self.nodes)))
            self.epss = torch.zeros((1, len(self.nodes)))
            self._epss_is_sampled = False
            self._obss_is_sampled = False
        elif len(sample_shape) == 1:
            self.obss = torch.zeros(sample_shape[0], len(self.nodes))
            self.epss = torch.zeros(sample_shape[0], len(self.nodes))
            self._epss_is_sampled = False
            self._obss_is_sampled = False
        else:
            raise ValueError("Sample shape must be a tuple of length 0 or 1")
        
    def set_eps(self, epss: torch.Tensor):
        assert len(epss.shape) == 2
        self.epss = epss
        self.obss = torch.zeros(epss.shape)
        self.abducted = False
        self.bounds = {}
        self._epss_is_sampled = True
        self._obss_is_sampled = False
        
    def epss_is_sampled(self):
        return self._epss_is_sampled
    
    def obss_is_sampled(self):
        return self._obss_is_sampled
        
    def copy(self):
        return copy.deepcopy(self)
    
    def shallow_copy(self):
        """
        Create a shallow copy of the SCM object.
        This copies the graph and the seqs, but not the obss and epss.
        """
        seqs = copy.deepcopy(list(self.seqs.values()))
        scm_copy = SCM(seqs)
        scm_copy.bounds = self.bounds.copy()
        scm_copy.abducted = self.abducted
        scm_copy.nodes = self.nodes.copy()
        scm_copy.node2idx = self.node2idx.copy()
        scm_copy.obss = self.obss
        scm_copy.epss = self.epss
        scm_copy._obss_is_sampled = self._obss_is_sampled
        scm_copy._epss_is_sampled = self._epss_is_sampled
        return scm_copy
        
    def do(self, int_dict: dict[str, float]) -> SCM:
        def make_const_fun(value):
            def f(par, eps):
                return torch.tensor([value]).float().repeat(eps.shape[0])
            return f
        scm_int = self.shallow_copy() # copies everything except obss and epss
        int_dict_safe = int_dict.copy()
        for node, val in int_dict_safe.items():
            f = make_const_fun(val)
            scm_int.seqs[node].f = f # type: ignore
            scm_int.seqs[node].f_inv = None # type: ignore
        return scm_int
    
    def abduct(self, obs: pd.Series) -> SCM:
        # Attention: We assume that obs is a single observation
        # TODO: think about whether this can be parallelized?
        # assert len(obs) == 1
        # assert type(obs) == pd.DataFrame
        # scm_abd = self.copy()
        # scm_abd.clear_sample()
        # for node in self.nodes:
        #     eps_node = self.seqs[node].inv(obs)
        #     scm_abd.seqs[node].d = Delta(eps_node, log_density=0.0)  
        # scm_abd.abducted = True          
        # return scm_abd
        raise NotImplementedError("Normal Abduction is not implemented yet")
    
    def abduct_one_unobserved(self, obs: pd.Series):        
        # find the unobserved node. We allow exactly one unobserved node
        unobs_nodes = [node for node in self.nodes if node not in obs.index]
        if len(unobs_nodes) != 1:
            raise ValueError("Exactly one node must be unobserved")
        unobs_node = unobs_nodes[0]
        
        # copy and clean the scm
        scm_abd = self.copy()
        scm_abd.clear_sample()
        
        
        # create a tensor for the observations with a zero for the unobserved node
        obs_safe = obs.copy()
        obs_safe[unobs_node] = 0.0
        obs_df = obs_safe.to_frame().T
        
        obss_tensor = torch.tensor(obs_df[self.nodes].to_numpy()).float()
        
        # nodes with standard abduction: all nodes except the unobserved node and its direct children
        special_abd = [unobs_node] + scm_abd.get_children(unobs_node)
        normal_abd = [n for n in scm_abd.nodes if n not in special_abd]
        for normal_node in normal_abd:
            eps_node = scm_abd.seqs[normal_node].inv(obss_tensor)
            eps_node = eps_node[0]
            scm_abd.seqs[normal_node].d = Delta(eps_node, log_density=0.0)
        
        # abduct the unobservered node
        dist_abd_unobs = DistAbdUnobs.from_scm(scm_abd, unobs_node, obs)
        scm_abd.seqs[unobs_node].d = dist_abd_unobs
        
        # model the noise of the children as a function of the unobserved node
        seq_unobs = scm_abd.seqs[unobs_node]
        unobs_ix = scm_abd.node2idx[unobs_node]
        
        # function factory to ensure early binding of the parameters
        def make_g_fn(obss_tensor: torch.Tensor, child: str, f_inv_ch: Callable, seq_unobs: Callable, ix_unobs: int) -> Callable[[torch.Tensor], torch.Tensor]:
            def g_fn(epss_unobs: torch.Tensor) -> torch.Tensor:
                """
                Computes the value of the unobserved node based on the noise of the unobserved node,
                and used that to determine the corresponding noise of the respective child node
                """
                assert child is not None # child variable only included for debugging purposes
                obss_tensor_local = obss_tensor.clone()
                obss_tensor_local = obss_tensor_local.repeat(epss_unobs.shape[0], 1)
                x_unobs_eps = seq_unobs(obss_tensor_local, epss_unobs)
                obss_tensor_local[:, ix_unobs] = x_unobs_eps
                return f_inv_ch(obss_tensor_local)                
            return g_fn
        
        # create corresponding noise for all children
        for child in scm_abd.get_children(unobs_node):
            seq = scm_abd.seqs[child]
            # use the function factory to create a function g_fn that maps U_Y to U_child
            g_fn = make_g_fn(obss_tensor, child, seq.inv, seq_unobs, unobs_ix)
            # use it to create the respective DependentNoiseSeq
            scm_abd.seqs[child] = DependentNoiseSeq(seq, g_fn, seq_dependent_var=seq_unobs) # type: ignore
        
        scm_abd.abducted = True
        return scm_abd
        
    def get_nodes_ordered(self) -> list[str]:
        return list(nx.topological_sort(self.graph))
    
    def get_parents(self, node: str) -> list[str]:
        return sorted(list(self.graph.predecessors(node)))
    
    def get_children(self, node: str) -> list[str]:
        return sorted(list(self.graph.successors(node)))
    
    def get_descendants(self, intv_set: list[str]) -> list[str]:
        desc_sets = [set(nx.descendants(self.graph, node)) for node in intv_set]
        if not desc_sets:
            # no nodes to process, so the union is empty
            return []
        desc_set = set.union(*desc_sets)
        desc_set = desc_set - set(intv_set)
        return list(desc_set)
    
    def get_ascendants(self, node_set: list[str]) -> list[str]:
        asc_sets = [set(nx.algorithms.dag.ancestors(self.graph, node)) for node in node_set]
        asc_set = set.union(*asc_sets)
        asc_set = asc_set - set(node_set)
        return list(asc_set)
    
    def get_nondescendants(self, intv_set: list[str]) -> list[str]:
        desc = set(self.get_descendants(intv_set))
        ndesc = set(self.nodes) - desc - set(intv_set)
        return list(ndesc)
    
    def get_sample(self, draw_sample: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        if draw_sample:
            self.sample()
        # convert the tensors to pandas dataframes
        if not self.epss_is_sampled() or not self.obss_is_sampled():
            raise RuntimeError("Sample not drawn yet. draw_sample must be True or sample() must be called before.")
        else:
            obss_df = pd.DataFrame(self.obss.numpy(), columns=self.nodes) # type: ignore
            epss_df = pd.DataFrame(self.epss.numpy(), columns=self.nodes) # type: ignore
        return obss_df, epss_df
    
    def sample_node(self, node: str, sample_shape: tuple | torch.Size=torch.Size([])):
        if type(sample_shape) is tuple:
            sample_shape = torch.Size(sample_shape)
        assert type(sample_shape) is torch.Size
        assert self.obss is not None
        assert self.epss is not None
        obss_node, epss_node = self.seqs[node].sample(self.obss, sample_shape=sample_shape, epss=self.epss)
        node_idx = self.node2idx[node]
        self.epss[:, node_idx] = epss_node.float()
        self.obss[:, node_idx] = obss_node.float()
        
    def sample(self, sample_shape: tuple | torch.Size=()) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert len(sample_shape) == 1
        if type(sample_shape) is tuple:
            sample_shape = torch.Size(sample_shape)
        assert type(sample_shape) is torch.Size
        self.clear_sample(sample_shape=sample_shape)
        for node in self.get_nodes_ordered():
            self.sample_node(node, sample_shape=sample_shape)
        self._epss_is_sampled = True
        self._obss_is_sampled = True
        return self.get_sample(draw_sample=False)
    
    def compute_node(self, node: str):
        """ Compute the values of the node without resampling the noise!"""
        assert self.obss is not None
        assert self.epss is not None
        assert node in self.nodes
        node_idx = self.node2idx[node]
        epss_node = self.epss[:, node_idx]
        obss_node = self.seqs[node](self.obss, epss_node)
        self.obss[:, node_idx] = obss_node
        
    def compute(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Computes the values of all nodes based on the current noise values,
        without resampling the noise and while ignorig all previous values."""
        assert self.epss_is_sampled()
        for node in self.get_nodes_ordered():
            self.compute_node(node)
        self._obss_is_sampled = True
        return self.get_sample(draw_sample=False)
            
    def log_prob(self, data: pd.DataFrame) -> np.ndarray:
        data_tensor = torch.tensor(data[self.nodes].to_numpy()).float()
        log_probs = [seq.log_prob(data_tensor) for seq in self.seqs.values()]
        log_probs_torch = torch.stack(log_probs, dim=1)
        log_probs_np = log_probs_torch.numpy()
        return np.sum(log_probs_np, axis=1)
    
    def get_bounds(self, sample_shape: tuple | torch.Size=(), force_recompute: bool=False):
        assert self.obss is not None
        if len(self.bounds) == 0 or force_recompute:
            # sample only what must be sampled to compute the bounds
            # sample shape is ignored if a sample is already available
            if not self.obss_is_sampled():
                if self.epss_is_sampled():
                    self.compute()
                else:
                    self.sample(sample_shape=sample_shape)
            for node in self.get_nodes_ordered():
                node_idx = self.node2idx[node]
                lb, ub = (self.obss[:, node_idx].min().item(), self.obss[:, node_idx].max().item())
                assert type(lb) is float
                assert type(ub) is float
                self.bounds[node] = (lb, ub)
        return self.bounds
    
    def get_support(self, node: str, sample_shape: tuple | torch.Size=(), force_recompute: bool=False) -> list:
        assert self.obss is not None
        if node not in self.support.keys() or force_recompute:
            # sample only what must be sampled to compute the bounds
            # sample shape is ignored if a sample is already available
            if not self.obss_is_sampled():
                if self.epss_is_sampled():
                    self.compute()
                else:
                    self.sample(sample_shape=sample_shape)
            for node_local in self.get_nodes_ordered():
                node_idx = self.node2idx[node_local]
                unique_vals = self.obss[:, node_idx].unique().tolist()
                self.support[node_local] = unique_vals
        return self.support[node]
    
    def get_description(self) -> str:
        desc = ""
        for node in self.get_nodes_ordered():
            desc += f"{node}: {self.seqs[node].description}\n"
        return desc
    
                