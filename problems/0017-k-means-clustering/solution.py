import numpy as np

def assign_centroid(points, centroids):
	#c_dict = {i:c for i, c in enumerate(centroids)}
	p2c_dists = {p: [0 for _ in range(len(centroids))] for p in points}
	c2p = {i:[] for i in range(len(centroids))}
	p2c = {}
	for p in points:
		for i, c in enumerate(centroids):
			dif = (p[0]-c[0], p[1]-c[1])
			p2c_dists[p][i] = np.linalg.norm(dif)
		p2c[p] = np.argmin(p2c_dists[p])
	for p, c in p2c.items():
		c2p[c].append(p)
	return c2p

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	centroids = initial_centroids
	while max_iterations > 0:
		c2p = assign_centroid(points, centroids)
		new_centroids = [tuple(np.mean(p, axis=0)) for _, p in c2p.items()]
		centroids = new_centroids
		max_iterations = max_iterations - 1
		
	final_centroids = new_centroids

	return final_centroids