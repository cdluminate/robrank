'''
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import faiss
import numpy as np
import argparse

ag = argparse.ArgumentParser()
ag.add_argument('--cuda', action='store_true')
ag.add_argument('--omp', type=int, default=4)
ag = ag.parse_args()
faiss.omp_set_num_threads(ag.omp)

N = 60502
C = 11316
D = 512

repres = np.random.randn(N, D).astype(np.float32)

cluster_idx = faiss.IndexFlatL2(D)
if ag.cuda:
    res = faiss.StandardGpuResources()
    cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
kmeans = faiss.Clustering(D, C)
kmeans.verbose = True
#kmeans.niter = 25
#kmeans.min_points_per_centroid = 1
#kmeans.max_points_per_centroid = 1000000
kmeans.train(repres, cluster_idx)

#kmeans = faiss.Kmeans(repres.shape[1], C, seed=123, verbose=True)
#kmeans.train(repres, cluster_idx)
#_, pred = kmeans.index.search(repres, 1)
_, pred = cluster_idx.search(repres, 1)
pred = pred.flatten()
print(pred)
