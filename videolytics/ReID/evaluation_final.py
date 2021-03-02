"""Evaluation script for evaluation of end-to-end approaches, that is final identities committed into database"""


from ReID import database_connection
from ReID.unionfind import UnionFind
from scipy.spatial import distance
import collections
import tqdm
import numpy as np
import datetime
import sqlalchemy.orm
import pickle
import heapq
import sys

OUR_MODEL_ID = int(sys.argv[1])
GOLDEN_MODEL_ID = 36

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


assignment_matrix = np.zeros((len(our_iden_idx_to_iden_id), len(gld_iden_idx_to_iden_id)), dtype=np.int64)
for det_id in det_id_to_our_iden_id.keys():
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

print("{:.4f} / {:.4f}".format(tp / (tp + fn), fp / (fp + tn)))
