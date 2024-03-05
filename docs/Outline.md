

The idea of this project is to leverage the parallelism of the newly presented [multi-fidelity reduced order surrogate modeling](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2023.0655) (this is the previous version I suppose) strategy in a computational inverse problem scenario.

The surrogate model has major relevance in a context where high fidelity simulations are computationally expensive and the time budget does not allow to use them. At the same time, we have the possibility to leverage less accurate, lower fidelity models that are computationally inexpensive.

### Multi Fidelity surrogate model
Algorithm (I have improvised a bit):

Suppose to have access to data $\mathbf{u}^{(f_k)}\in\mathbb{R}^d$ characterized by different level of fidelities $f_k$ where $f_0=LF$ is the lowest fidelity available and $f_{N_f}=HF$ is the highest fidelity. 
Each data is characterized by a cost $c_f$ to obtain it s.t. $0=c_0<c_1<...<c_{N_f}$ . Then:

1. Generate training data using $N_{sim}$ high fidelity simulations $\mathbf{u}^{(HF)}_k$ for $k=1,...,N_{sim}$
2. Train the lowest fidelity surrogate model $\phi^{LF} : \mathbb{R}^d \mapsto \mathbf{u}^{HF}$. The low fidelity model is parametrized by a set of coefficients $\theta^{LF}$ which are found by solving:$$\theta^{LF}=\arg\min_{\theta^{LF}}\sum_{k=1}^{N_{sim}}\|u^{(HF)}_k - \phi^{LF}(\mathbf{u}^{LF}_k;\theta^{LF})\|$$ for a given norm and fix the values of $\theta^{LF}$ 
3. For $j = 1 : N_f-1$ do:
	- Sub-select a portion of reference simulations $N_j$ from the training reference data
	- Build an increasing fidelity surrogate model $\phi^{f_j}$ parametrized by $\theta^{f_j}$ and $\phi^{f_j}:\mathbf{u}^{f_j} \mapsto \mathbb{R}^d$ 
	- Find the values $\theta^{f_j}$ by solving:$$\theta^{f_j}=\arg\min_{\theta^{f_j}}\sum_{k=1}^{N_{j}}\left\|u^{(HF)}_k - \sum_{l=0}^{j}\phi^{f_l}(\mathbf{u}^{f_l}_k;\theta^{f_l})\right\|$$
The Multi fidelity surrogate model of level $L$ is given by the following:
$$\phi^{L}_{MF}(\mathbf{u}^{f_0},\ldots,\mathbf{u}^{f_L}) = \sum_{j=0}^L\phi^{f_j}(\mathbf{u}^{f_j},\theta^{f_j}) $$

With a maximum level $L_{MAX}=N_f-1$.

The idea is to leverage this increasing level of multi-fidelity surrogate model in a multi-level delayed acceptance Monte-Carlo algorithm:


### Multi-Level Delayed acceptance Monte Carlo (adjusted)

Described [here](https://arxiv.org/pdf/2202.03876.pdf) 

Each Multi fidelity surrogate model of level $l=1:L_{MAX}$ is associated to a target density $\pi_l$.

The samples $\theta^1 ... \theta^N$ are obtained using the following algorithm:

1. Set $L=0$ 
2. for $j=0...N-1$ do:
	- for $l=0:L$
		- Sample a number of steps $n_l$
		- Generate the level $l$ data $\mathbf{u}^{f_l}$
		- Perform a Metropolis Hastings chain of length $n_l$ using the surrogate model $\phi^{l}_{MF}(\mathbf{u}^{f_0},\ldots,\mathbf{u}^{f_l})$ starting from $\theta^{l-1}$
		- Take the last sample as $\theta^l$
		take $\theta^L$ as the j-th sample and use it to start the MH chain of level 0 of next step.


### Application


Parameter identification in the context of 1d MEMS accelerometer.

Idea: use this case as a first verification case, since an identification routine based on data-driven models has already been established.

Parameters: over-etch, offset, thickness, quality factor
States: displacement field, velocity field, electric field

Surrogate Models for increasing fidelity:

0. Highest Fidelity, FEM multi-physics simulation. Idea: use FENICS in order to be intrusive if needed (e.g. mesh rearrangement) and facilitate application to different scenarios
1. Lowest Fidelity, Parameters to solution map (0 costs)
2. Medium Fidelity, use a 1D analytical model approximation of a MEMS accelerometer (ODE), solve the equation using a suitable Euler Scheme and use it as input (with parameters) for the surrogate model. 1D signal $\rightarrow$ LSTM ?
3. Higher Fidelity: coarse FEM simulation of the displacement of the accelerometer (+ analytical formulas for the electric field?) or Euler-Bernoulli approximations (Coventor MEMS+) $\rightarrow$ POD-NN, POD+LSTM, DL-ROM

Criticalities:
1. For the present case probably a 1D analytical model is already enough to have a very powerful approximation. Consider to move to more complex scenarios to prove the importance of a multi-fidelity strategy
2. For the parameter Identification a quantity of interest based on the solution (capacitance variance) is enough, making the map easier to be learned. A more interesting application would be using a full field solution as an output (e.g. Structural Health Monitoring scenarios), but this can be a good starting point
3. Show the effectiveness of using the present strategy w.r.t. different levels of accuracy or other multi-fidelity techniques that blend information

Interesting Results:
1. Check that the implemented method preserves the same accuracy of the Developed Routine for the computational inverse problem
2. Perform a cost effective analysis and compare the efficiency with respect to using a single level MCMC
3. Find an optimal strategy for the Time Budget expenses

Possible Developments:
1. Other Scenarios
2. Online learning. During the training, points that are outside of the training dataset range may be reached. Consider whether to update the model parameters using an effective online learning strategy when this happens
3. Adaptivity. A powerful improvement would be to have that the lower fidelities model further adapt and improve while the chain is evolving (e.g. read [here](https://arxiv.org/abs/1403.4290)), as this would make the routine extremely more efficient.
4. Use a horizontal instead of a parallel structure, we can leverage the 

Multi-fidelity reference:
- [Willcox survey](https://epubs.siam.org/doi/10.1137/16M1082469)
- lavoro a monte e tipo di sampler


