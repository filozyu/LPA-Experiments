# LPA-Experiments

Check: (Fast RST):
1. https://giovannizappella.github.io/papers/jwta.pdf
2. https://github.com/ermeel86/wilsons_algorithm_in_python
3. Test on lollipop graphs


TO-DO List 04 Mar 2019

- Experiments:
	1. Ones vs. Twos in NN graph (1-NN, 3-NN and 10-NN)
	2. Odd number of classifiers (or spines) (3,5,7,9 etc.)
	3. For 20newsgroups: remove edges based on the weights
	

TO-DO List 01 Mar 2019

- Experiments:
	1. LPA on the original graph
	2. Random spanning tree then linear embedding (spine) (ensemble: 1,3,5,7,15,31) . One spine per tree, LPA on the spines.
	3. Minimal spanning tree LPA

- LPA options: assign labels according to
	1. K Nearest Neighbour (explore different choices of K)
	2. LPA implemented in GAMMA (Gaussian field harmonic functions)

- RST and spine:
	1. Use random walk to generate RSTs
	2. For each RST, use depth first search to generate a spine (drop duplicates in a fixed way, like drop all but the first occurrence)
	3. Random sample labels from the spines
	4. Propagate labels on each spine using LPA
	5. Ensemble the results on the spines
