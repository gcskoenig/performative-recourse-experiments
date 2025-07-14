from __future__ import annotations
import torch
import torch.distributions as dist
from torch.distributions.constraints import Constraint
import pandas as pd
import copy
from typing import Callable, Optional
import logging
from torch.distributions.utils import broadcast_all
from torch.distributions import constraints


logger = logging.getLogger(__name__)

def discretize_dist(d: dist.Distribution, num_samples: int = 1000) -> dist.Distribution:
    # # 1. Draw samples from the continuous distribution
    # samples = d.sample((num_samples,))

    # # 2. Round or bin the samples
    # discrete_samples = torch.round(samples)

    # # 3. Get the unique discrete values and counts
    # bin_values, bin_counts = discrete_samples.unique(return_counts=True)
    
    # # 4. Convert counts to a probability distribution
    # pmf = bin_counts.float() / bin_counts.sum()

    # # 5. Create a Categorical distribution
    # discrete_dist = dist.Categorical(probs=pmf)

    # return discrete_dist
    raise NotImplementedError("Discretization is not implemented yet, issues with the support")


class SafeLogProbWrapper(dist.Distribution):
    arg_constraints = {} # type: ignore
    has_rsample = False
    
    def __init__(self, base_dist: dist.Distribution, validate_args=None):
        super().__init__(batch_shape=base_dist.batch_shape,
                         event_shape=base_dist.event_shape,
                         validate_args=validate_args)
        self.base_dist = base_dist
        self.safe_val = base_dist.sample((1,))  # Dummy value to avoid errors in log_prob

    def log_prob(self, value):
        mask = self.support.check(value) # type: ignore
        # Create a dummy value in-support to safely call log_prob
        # Expand mask to (n, d) so we can zero out the entire row for out-of-support entries
        expanded_mask = mask.unsqueeze(-1).expand_as(value)

        # Create a dummy in-support value (just 0 for rows that are out-of-support)
        safe_value = torch.where(expanded_mask, value, torch.ones_like(value)*self.safe_val)
        
        # round very close to int values to the nearest int
        int_mask = torch.isclose(safe_value, torch.round(safe_value), atol=1e-3)
        safe_value[int_mask] = torch.round(safe_value[int_mask]).int().float()

        # Now compute log_prob on these "safe" rows
        logp = self.base_dist.log_prob(safe_value)  # Typically returns shape (n,) for event_shape=(d,)

        # Finally, assign -inf to rows that were out-of-support
        return torch.where(mask, logp, torch.tensor(-float('inf'), device=value.device, dtype=logp.dtype))

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)
    
    @property
    def support(self):
        # Directly delegate to base_dist
        return self.base_dist.support

    def __getattr__(self, name):
        # Forward any other attributes to the base distribution
        return getattr(self.base_dist, name)


class ProductConstraint(Constraint):
    """
    A constraint that is satisfied only if each dimension i of `value`
    is in the i-th child constraint.
    """
    def __init__(self, constraint_list):
        super().__init__()
        self.constraint_list = constraint_list  # e.g. [Normal(0,1).support, Uniform(0,1).support, ...]

    def check(self, value):
        """
        value shape: (..., n_dims)
        
        We check dimension i using constraint_list[i].
        """
        # Each constraint_list[i].check(...) returns a boolean tensor of shape (...).
        # We'll accumulate them via logical AND.
        checks = []
        for i, c in enumerate(self.constraint_list):
            # Take the i-th slice of the last dimension
            x_i = value[..., i]
            checks.append(c.check(x_i))

        # Combine them with logical AND
        final_check = checks[0]
        for chk in checks[1:]:
            final_check = final_check & chk
        return final_check

class JointDistribution(dist.Distribution):
    arg_constraints = {} # type: ignore
    
    def __init__(self, distributions: list[dist.Distribution]):
        """
        Create a joint distribution object from a list of independent distributions.
        
        :param distributions: List of independent distribution objects (e.g., Normal, Uniform, etc.)
        """
        self.distributions = distributions
        
        batch_shapes = [d.batch_shape for d in self.distributions]
        combined_batch_shape = torch.broadcast_shapes(*batch_shapes) if batch_shapes else ()
        
        combined_event_shape = (len(self.distributions),)
        
        super().__init__(
            batch_shape=torch.Size(combined_batch_shape), 
            event_shape=torch.Size(combined_event_shape)
        )

    def sample(self, sample_shape: torch.Size | tuple = torch.Size()) -> torch.Tensor:
        """
        Samples from the joint distribution by sampling from each individual distribution.
        
        :param sample_shape: Shape of the sample to be generated.
        :return: A tensor containing the joint sample.
        """
        samples = [d.sample(sample_shape) for d in self.distributions]
        samples = torch.stack(samples, dim=-1)
        assert samples.shape == sample_shape + (len(self.distributions),)
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability of the given value under the joint distribution.
        
        :param value: A tensor containing the values for which to compute the log-probability.
        :return: Log-probability tensor for the given value.
        """
        log_prob_sum = torch.zeros(value.shape[0])
        for i, d in enumerate(self.distributions):
            log_prob_i = d.log_prob(value.select(dim=-1, index=i))
            log_prob_sum += log_prob_i        
        return log_prob_sum
    
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def support(self):
        # Gather child supports
        child_supports = [d.support for d in self.distributions]
        # Return a product constraint
        return ProductConstraint(child_supports)

    def enumerate_support(self, expand: bool = True) -> torch.Tensor:
        """
        Enumerates the support of the joint distribution by combining the supports
        of the individual distributions.
        
        :param expand: Whether to expand the enumeration to match the batch shape.
        :return: A tensor containing the enumerated support.
        """
        supports = [d.enumerate_support(expand=expand) for d in self.distributions]
        # Create a Cartesian product of the supports
        grids = torch.meshgrid(*supports, indexing='ij')
        enumerated_support = torch.stack([g.flatten() for g in grids], dim=-1)
        return enumerated_support

class DistAbdUnobs(dist.Distribution):
    r"""
    Custom distribution defined on the y-space whose density is proportional to
      p(y) ∝ p_{D_Y}(y) * p_{D_E}(g(y))
    where g is an invertible function.
    
    The normalized density is:
      p(y) = [p_{D_Y}(y) * p_{D_E}(g(y))] / C
    with C being the normalizing constant.
    
    Sampling is implemented using rejection sampling. We use D_Y as the proposal.
    Each candidate y ~ D_Y is accepted with probability:
         accept_prob = p_{D_E}(g(y)) / M,
    where M is a constant satisfying:
         p_{D_E}(g(y)) ≤ M  for all y.
    """
    # (You can set constraints here as needed.)
    arg_constraints = {} # type: ignore
    support = dist.constraints.real # type: ignore

    def __init__(self, d_y: dist.Distribution, d_e: dist.Distribution,
                 g_fn: Callable[[torch.Tensor], torch.Tensor], M: float,
                 log_C: Optional[float] = None, validate_args = None):
        """
        Args:
            d_y (Distribution): PyTorch distribution for D_Y.
            d_e (Distribution): PyTorch distribution for D_E.
            g_fn (callable): A function mapping y -> e (should be invertible).
            M (float): A constant such that for all y, p_{D_E}(g(y)) ≤ M.
            log_C (float, optional): The log normalizing constant. If provided,
                                     log_prob returns a normalized density.
                                     Otherwise, log_prob returns an unnormalized value.
        """
        super().__init__(validate_args=validate_args)
        self.d_y = d_y
        self.d_e = d_e
        self.g_fn = g_fn
        self.M = M
        self.log_C = log_C
        
        # Override the instance's constraints and support with those from d_y:
        self.arg_constraints = getattr(d_y, "arg_constraints", {}) # type: ignore
        self.support = getattr(d_y, "support", dist.constraints.real) # type: ignore
    
    @classmethod
    def from_scm(cls, scm: SCM, unobserved_node: str, obs: pd.Series | dict, # type: ignore
                 M: Optional[float] = None):
        """
        Create a DistAbdUnobs object from a SCM and an observation.
        
        Args:
            scm (SCM): Structural causal model.
            unobserved_node (str): Name of the unobserved node for which to compute the distribution.
            obs (dict): Observation of the SCM where exactly one node is unobserved.
            M (float): A constant such that for all y, p_{D_E}(g(y)) ≤ M.
        
        Returns:
            A DistAbdUnobs object with the given observation.
        """        
        sorted_children = scm.get_children(unobserved_node)
        if len(sorted_children) == 0:
            raise ValueError("The unobserved node must have at least one child")

        # product distirbution of the children
        d_e = JointDistribution([scm.seqs[child].d for child in sorted_children])
        
        if type(obs) is pd.Series or type(obs) is dict:
            if type(obs) is dict:
                obs_df = pd.Series(obs).to_frame().T
            elif type(obs) is pd.Series:
                obs_df = obs.to_frame().T
        else:
            raise ValueError("Observation must be a dictionary or pd.Series")
        
        if obs_df.isna().any().any():
            raise ValueError("Observation must not contain missing values")
        
        # approximate M by sampling
        if M is None:
            smpl = d_e.sample(torch.Size([10**4]))
            M = torch.max(d_e.log_prob(smpl)).item() * 1.01            
        
        # copying variables to avoid side effects inside the closure
        obs_original = obs_df.copy(deep=True)
        scm_original = scm.copy()
        unobserved_node_original = str(unobserved_node)
        sorted_children_original = sorted_children.copy()
        d_e_original = d_e.copy()
        M_original = float(M)

        # a function that meps from eps_unobserved to eps_children
        def g_fn(eps_unobserved: torch.Tensor) -> torch.Tensor:
            obs_local = copy.deepcopy(obs_original)
            obs_local[unobserved_node_original] = 0.0
            obs_local = torch.tensor(obs_local[scm.nodes].to_numpy()).reshape(-1, len(scm.nodes)).float()
            
            # create a corresponding observation array filling in the known values
            if obs_local.shape[0] == 1:
                obs_local = obs_local.repeat(eps_unobserved.shape[0], 1)
            
            # use the eps_unobserved to compute the observed value for the node
            x_unobserved = scm_original.seqs[unobserved_node_original](obs_local, eps_unobserved)
            unobserved_ix = scm.node2idx[unobserved_node_original]
            obs_local[:, unobserved_ix] = x_unobserved
            
            # use the inverse function for the children to compute the eps for the children
            eps_children = []
            for child in sorted_children_original:
                f_inv = scm_original.seqs[child].inv
                eps_children.append(f_inv(obs_local))
                # assert that computed values are correct
                obs_child_new = scm_original.seqs[child](obs_local, eps_children[-1])
                assert torch.allclose(obs_child_new, obs_local[:, scm.node2idx[child]], atol=1e-3), \
                    f"Computed value for child {child} does not match the original observation. "
                
            # concatenate the eps for the children and return as torch tensor
            # make sure that the result is a 2D tensor
            eps_children = torch.stack(eps_children, dim=1)
            assert eps_children.shape == (eps_unobserved.shape[0], len(sorted_children_original))
            
            return eps_children           
            
        d = DistAbdUnobs(scm_original.seqs[unobserved_node_original].d,
                         d_e_original, g_fn, M_original)
        return d

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at y:
          log p(y) = log p_{D_Y}(y) + log p_{D_E}(g(y)) - log_C   (if log_C is provided)
                   = log p(y) = log p_{D_Y}(y) + log p_{D_E}(g(y))            (if log_C is None)
        """
        logp_y = self.d_y.log_prob(y)
        logp_e = self.d_e.log_prob(self.g_fn(y))
        if self.log_C is None:
            return logp_y + logp_e  # unnormalized density
        else:
            return logp_y + logp_e - self.log_C

    def sample(self, sample_shape=torch.Size(), sampling_batch_size=10**4, max_tries=100):
        """
        Draws samples from the distribution using rejection sampling.
        
        The procedure is:
          1. Draw a candidate y from D_Y.
          2. Compute acceptance probability: p_{D_E}(g(y)) / M.
          3. Accept or reject the candidate based on a Uniform(0,1) draw.
        
        Args:
            sample_shape (torch.Size): Desired sample shape.
        
        Returns:
            A tensor of accepted samples with shape sample_shape.
        """
        # Determine the total number of samples needed.
        num_samples = int(torch.tensor(sample_shape).prod().item()) if sample_shape else 1
        accepted_samples = torch.tensor([]) # to store accepted candidates
        
        running_acc_prob = 0.0
        tries = 0
        
        # Keep generating candidate batches until we have enough samples.
        while len(accepted_samples) < num_samples:
            # Draw a batch of candidates from D_Y.
            candidates = self.d_y.sample(sample_shape=(sampling_batch_size,))
            # Evaluate log density of D_E at g(candidates) and convert to probability.
            d_e_safe = SafeLogProbWrapper(self.d_e)
            g_fn_cands = self.g_fn(candidates)
            log_p_e = d_e_safe.log_prob(g_fn_cands)
            # Compute acceptance probabilities.
            accept_log_prob =  log_p_e - self.M
            accept_prob = torch.exp(accept_log_prob)
            running_acc_prob += torch.sum(accept_prob).item()
            
            # assert that accept_prob is between 0 and 1
            logger.debug(f"Maximum acceptance probability: {torch.max(accept_prob)}")
            
            # Draw uniform random numbers for the rejection test.
            u = torch.rand(accept_log_prob.shape)
            
            # Create a boolean mask for accepted candidates.
            mask = u < torch.exp(accept_log_prob)
            # Extract accepted candidates.
            accepted = candidates[mask]
            accepted_samples = torch.cat([accepted_samples, accepted], dim=0)
            tries += 1
            
            if tries == max_tries and running_acc_prob == 0:
                pass # for debugging purposes
            elif tries > max_tries and running_acc_prob == 0:
                error_msg = f"Zero acceptance probability over {tries} rounds. Check the proposal distribution and acceptance criteria."
                error_msg += f"Candidates: {torch.unique(candidates, return_counts=True)}"
                error_msg += f"g_fn_candidates: {torch.unique(g_fn_cands, return_counts=True)}"
                raise ValueError(error_msg)
            elif tries > max_tries * 50:
                raise ValueError(f"Too many tries ({tries}) to get enough samples. Consider increasing sampling_batch_size or decreasing num_samples.")
        
        # Concatenate all accepted candidates and trim to the desired number.
        accepted_samples = accepted_samples[:num_samples]
        # Reshape to the requested sample_shape (preserving extra dimensions from the candidate).
        return accepted_samples.reshape(sample_shape).float()

class ShiftedBinomialConstraint(constraints.Constraint):
    r"""
    Constrains values to be in the set { offset, offset+1, ..., offset+n } 
    for each batch element. Equivalently, (value - offset) must be an integer 
    in the interval [0, n].
    """

    def __init__(self, offset, n):
        super().__init__()
        self.offset = offset
        self.n = n

    def check(self, value):
        # Because offset and n may be tensors, broadcast everything to a common shape
        offset, n, value = broadcast_all(self.offset, self.n, value)
        x = value - offset
        # 1) x is between 0 and n
        # 2) x is an integer
        is_integer = torch.isclose(x, torch.round(x), atol=1e-3)        
        if torch.any(~is_integer):
            logger.info("Some values checked are not integers (ShiftedBionomialConstraint). There might be a bug.")
        x[is_integer] = torch.round(x[is_integer]).int().float()
        in_range = (x >= 0) & (x <= n)
        return in_range & is_integer
        
class ShiftedBinomial(dist.Distribution):
    r"""
    A "binomial-like" distribution with parameters (n, p), shifted by (mu - n*p) 
    so that its mean is mu. 

    The unshifted distribution is Binomial(total_count=n, probs=p). 
    We define offset = (mu - n*p). We then:
      - sample from Binomial, 
      - shift by offset, 
      - and define the log_prob appropriately.

    The support of this distribution is discrete points 
      offset + {0, 1, 2, ..., n}.

    Args:
        n (int or Tensor): number of trials
        p (float or Tensor): success probability
        mu (float or Tensor): desired mean (so offset = mu - n*p)
    """

    arg_constraints = { # type: ignore
        'n': constraints.positive_integer,
        'p': constraints.unit_interval,
        'mu': constraints.real,
    }
    # We'll define the (discrete) support dynamically, using a custom constraint
    has_enumerate_support = False

    def __init__(self, n, p, mu, validate_args=None):
        self.n, self.p, self.mu = broadcast_all(n, p, mu)

        # Base Binomial distribution
        self.binom = dist.Binomial(total_count=self.n, probs=self.p)
        # The shift that ensures mean = mu
        self.offset = self.mu - self.n * self.p
        
        # Standard init
        super().__init__(
            batch_shape=self.binom.batch_shape,
            event_shape=torch.Size(),
            validate_args=validate_args
        )

        # Attach a custom constraint for discrete support
        self._support = ShiftedBinomialConstraint(self.offset, self.n)
        
        if not torch.all(self.offset == torch.floor(self.offset)):
            raise ValueError("Offset must be an integer (mu - n*p). Only specify p such that n*p is an integer or choose mu with an according float.")


    @property
    def support(self):
        # Return our custom constraint 
        return self._support

    @property
    def mean(self):
        # By design, the mean is mu
        return self.mu

    @property
    def variance(self):
        # Shifting does not change the variance of the base distribution
        return self.binom.variance
    
    def enumerate_support(self, expand: bool = True) -> torch.Tensor:
        return self.binom.enumerate_support(expand) + self.offset

    def sample(self, sample_shape=torch.Size()):
        # Sample from the underlying Binomial, then shift
        x = self.binom.sample(sample_shape)
        return x + self.offset

    def rsample(self, sample_shape=torch.Size()):
        # Binomial is not reparameterizable, so just sample
        return self.sample(sample_shape)

    def log_prob(self, value):
        # Convert real-valued samples back to the base binomial domain
        x = value - self.offset

        # Must be integer and 0 <= x <= n
        is_integer = torch.isclose(x, torch.round(x), atol=1e-3)
        x[is_integer] = torch.round(x[is_integer]).int().float()
        in_range = (x >= 0) & (x <= self.n) & is_integer

        # Compute base binomial log pmf for valid x; else -inf
        try:
            log_base = self.binom.log_prob(x)
        except Exception as e:
            logger.error(f"Error in log_prob computation: {e}")
            raise e
        return torch.where(in_range, log_base, torch.full_like(value, float('-inf')))