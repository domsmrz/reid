"""Evaluation script for evaluation of trajectory merging strategies"""


from . import database_connection
from .unionfind import UnionFind
from scipy.spatial import distance
import collections
import tqdm
import numpy as np
import datetime
import sqlalchemy.orm
import pickle
import heapq
import sys

REPR_DIST = 0.2
OUR_MODEL_ID = 118
GOLDEN_MODEL_ID = 36
FEATURE_MODEL_ID = 811

max_time = (
        database_connection.session.query(sqlalchemy.func.min(database_connection.Frame.timestamp))
        .filter(database_connection.Frame.camera_id.in_([467, 468]))
        .scalar() + datetime.timedelta(minutes=2, seconds=5)
)

golden = (
    [(did, iid) for did, iid, ts in
     database_connection.session.query(database_connection.Detection.id, database_connection.Identity.id, database_connection.Frame.timestamp)
         .select_from(database_connection.Identity)
         .join(database_connection.IdentityDetection)
         .join(database_connection.Detection)
         .join(database_connection.Frame)
         .filter(database_connection.Identity.identity_model_id == GOLDEN_MODEL_ID)
         .all() if ts < max_time]
)
golden_det_ids_set = {d for d, _ in golden}
our = (
    [(did, iid) for did, iid, ts in
     database_connection.session.query(database_connection.Detection.id, database_connection.Identity.id, database_connection.Frame.timestamp)
     .select_from(database_connection.Identity)
     .join(database_connection.IdentityDetection)
     .join(database_connection.Detection)
     .join(database_connection.Frame)
     .filter(database_connection.Identity.identity_model_id == OUR_MODEL_ID)
     .all() if ts < max_time and did in golden_det_ids_set]
)
data = (
    [(did, iid) for did, iid, ts in
     database_connection.session.query(database_connection.Detection.id, database_connection.FeatureDescriptor.value, database_connection.Frame.timestamp)
     .join(database_connection.Frame)
     .join(database_connection.FeatureDescriptor,
           (database_connection.Detection.id == database_connection.FeatureDescriptor.detection_id) & (database_connection.FeatureDescriptor.feature_type_id == FEATURE_MODEL_ID)
           )
     .order_by(database_connection.Detection.id)
     .all() if ts < max_time and did in golden_det_ids_set]
)

det_idx_to_features = [database_connection.np.frombuffer(x, dtype=database_connection.np.float32) for _, x in data]
det_idx_to_det_id = [x for x, _ in data]
det_id_to_features = dict(zip(det_idx_to_det_id, det_idx_to_features))
det_id_to_our_iden_id = dict(our)
det_id_to_gld_iden_id = dict(golden)

our_iden_id_to_det_ids = collections.defaultdict(list)
for did, iid in det_id_to_our_iden_id.items():
    our_iden_id_to_det_ids[iid].append(did)

gld_iden_id_to_det_ids = collections.defaultdict(list)
for did, iid in det_id_to_gld_iden_id.items():
    gld_iden_id_to_det_ids[iid].append(did)

our_iden_idx_to_iden_id = sorted(our_iden_id_to_det_ids.keys())
gld_iden_idx_to_iden_id = sorted(gld_iden_id_to_det_ids.keys())
our_iden_id_to_iden_idx = {x: i for i, x in enumerate(our_iden_idx_to_iden_id)}
gld_iden_id_to_iden_idx = {x: i for i, x in enumerate(gld_iden_idx_to_iden_id)}

our_iden_id_to_rep_ids = dict()
for iid, dids in our_iden_id_to_det_ids.items():
    v = list()
    for did in dids:
        if all(distance.cosine(det_id_to_features[did], det_id_to_features[x]) > REPR_DIST for x in v):
            v.append(did)
    our_iden_id_to_rep_ids[iid] = v

rep_idx_to_rep_id = [x for y in our_iden_id_to_rep_ids.values() for x in y]

heap = list()
for i, x in tqdm.tqdm(enumerate(rep_idx_to_rep_id), total=len(rep_idx_to_rep_id)):
    for j in range(i+1, len(rep_idx_to_rep_id)):
        d = distance.cosine(det_id_to_features[x], det_id_to_features[rep_idx_to_rep_id[j]])
        # rep_dist_mat[i,j] = rep_dist_mat[j,i] = d
        heap.append((d, i, j))

print("Heapifying", file=sys.stderr)
heapq.heapify(heap)
print("Done Heapifying", file=sys.stderr)

assignment_matrix = np.zeros((len(our_iden_idx_to_iden_id), len(gld_iden_idx_to_iden_id)), dtype=np.int64)
for det_id in det_idx_to_det_id:
    o = our_iden_id_to_iden_idx[det_id_to_our_iden_id[det_id]]
    g = gld_iden_id_to_iden_idx[det_id_to_gld_iden_id[det_id]]
    assignment_matrix[o, g] += 1
sums_per_our = assignment_matrix.sum(axis=1)
sums_per_gld = assignment_matrix.sum(axis=0)
total_sum = assignment_matrix.sum()

tp = fp = fn = tn = 0
for i in range(len(our_iden_idx_to_iden_id)):
    for j in range(len(gld_iden_idx_to_iden_id)):
        x = assignment_matrix[i,j]
        tp += x * (x-1)
        fp += x * (sums_per_our[i] - x)
        fn += x * (sums_per_gld[j] - x)
        tn += x * (total_sum - sums_per_our[i] - sums_per_gld[j] + x)

current_iden_idx_to_det_ids = UnionFind(det_idx_to_det_id)
for iden_id, det_ids in our_iden_id_to_det_ids.items():
    it = iter(det_ids)
    head = next(it)
    for i in it:
        current_iden_idx_to_det_ids.union(head, i)

it = tqdm.tqdm(total=len(current_iden_idx_to_det_ids.roots))
while len(current_iden_idx_to_det_ids.roots) > 1:
    d, *idx = heapq.heappop(heap)
    det_ids = tuple(rep_idx_to_rep_id[x] for x in idx)
    roots = tuple(current_iden_idx_to_det_ids.find(x) for x in det_ids)
    if roots[0] is roots[1]:
        continue
    it.update(1)
    our_iden_idxs = [det_id_to_our_iden_id[x] for x in det_ids]
    preserved_root = current_iden_idx_to_det_ids.union(*det_ids)
    deleted_root = roots[0] if preserved_root is roots[1] else roots[1]
    preserved_our_iden_idx = our_iden_id_to_iden_idx[det_id_to_our_iden_id[preserved_root]]
    deleted_our_iden_idx = our_iden_id_to_iden_idx[det_id_to_our_iden_id[deleted_root]]

    for j in range(len(gld_iden_idx_to_iden_id)):
        i1, i2 = [deleted_our_iden_idx, preserved_our_iden_idx]
        x, y = assignment_matrix[i1, j], assignment_matrix[i2, j]
        s = x + y
        tp += s * (s - 1) - x * (x - 1) - y * (y - 1)
        fp += s * (sums_per_our[i1] + sums_per_our[i2] - s) - x * (sums_per_our[i1] - x) - y * (sums_per_our[i2] - y)
        fn += s * (sums_per_gld[j] - s) - x * (sums_per_gld[j] - x) - y * (sums_per_gld[j] - y)
        tn += (
            s * (total_sum - sums_per_our[i1] - sums_per_our[i2] - sums_per_gld[j] + s)
            - x * (total_sum - sums_per_our[i1] - sums_per_gld[j] + x)
            - y * (total_sum - sums_per_our[i2] - sums_per_gld[j] + y)
        )

    assignment_matrix[preserved_our_iden_idx, :] += assignment_matrix[deleted_our_iden_idx,:]
    sums_per_our[preserved_our_iden_idx] += sums_per_our[deleted_our_iden_idx]
    assignment_matrix[deleted_our_iden_idx, :] = 0
    sums_per_our[deleted_our_iden_idx] = 0
    print(tp / (tp + fn), fp / (fp + tn), *det_ids, d)

