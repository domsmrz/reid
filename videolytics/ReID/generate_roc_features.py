"""Evaluation file for generating ROC curves for annotators"""


from .database_connection import *
import plotly.graph_objects as go
from .utils import *
import sqlalchemy.orm
import datetime
import collections
import random
from scipy.spatial import distance
from fractions import Fraction
import tqdm
from sklearn import metrics
import math
import sys

N = 50000
CAMERA_IDS = 467, 468
random.seed(42)
TIME_OFFSET = datetime.timedelta(minutes=2, seconds=5)

feature_type_ids = list()
names = list()

for x in sys.argv[2:]:
    if ',' in x:
        id_, name = x.split(',', 1)
        feature_type_ids.append(int(id_))
        names.append(name)
    else:
        feature_type_ids.append(int(x))
        names.append(x)

max_time = session.query(sqlalchemy.func.min(Frame.timestamp)).filter(Frame.camera_id.in_(*CAMERA_IDS)).scalar() + TIME_OFFSET

fvs = [sqlalchemy.orm.aliased(FeatureDescriptor) for _ in feature_type_ids]

query = (
    session.query(Identity.id, Frame.timestamp, Frame.camera_id, *(x.value for x in fvs))
    .select_from(IdentityDetection)
    .join(Identity)
    .join(Detection)
    .join(Frame)
    .filter(Identity.identity_model_id == 36)
    .filter(Frame.timestamp < max_time)
    .filter(Frame.camera_id.in_([467, 468]))
    .filter(Detection.class_ == 'PERSON')
    .order_by(IdentityDetection.id)
)

for fv, fvid in zip(fvs, feature_type_ids):
    query = query.join(fv, (fv.detection_id == Detection.id) & (fv.feature_type_id == fvid))

detections = collections.defaultdict(list)
for identitiy_id, *data in query:
    detections[identitiy_id].append(data)

to_del = [k for k, v in detections.items() if len(v) < 5]
for x in to_del:
    del detections[x]

det_val = list(detections.values())

sames = list()
diffs = list()

min_time_dist = datetime.timedelta(seconds=2)

triplets = list()
while len(triplets) < N:
    traj1, traj2 = random.sample(det_val, 2)
    (a_time, a_camera, *a_data), (p_time, p_camera, *p_data) = random.sample(traj1, 2)
    if a_camera == p_camera and abs(a_time - p_time) < min_time_dist:
        continue
    n_time, n_camera, *n_data = random.choice(traj2)
    data = [[np.frombuffer(x, dtype=np.float32) for x in y] for y in [a_data, p_data, n_data]]
    triplets.append(data)


fig = go.Figure()

dist = distance.cityblock
dist = getattr(distance, sys.argv[1])
if True:
  for i, (fid, name) in enumerate(zip(feature_type_ids, names)):
    ft = session.query(FeatureType).filter_by(id=fid).one()
    computed_distances = [
        (
            dist(x[0][i], x[1][i]),
            dist(x[0][i], x[2][i]),
        ) for x in triplets
    ]

    computed_distances = [
        ((math.inf if math.isnan(x) else x), (math.inf if math.isnan(y) else y))
        for x, y in computed_distances
    ]

    computed_distances = [((math.inf, y) if math.isnan(x) else (x, y)) for x, y in computed_distances]

    loss = sum(max(same - diff + 1, 0) for same, diff in computed_distances) / N

    raw_data = list()
    for same, diff in computed_distances:
        raw_data.append((same, 1))
        raw_data.append((diff, 0))

    raw_data.sort()

    tpr = list()
    fpr = list()

    tp = 0
    tn = N
    fp = 0
    fn = N

    tpr.append(0)
    fpr.append(0)

    last_dist = -1
    for this_dist, cls in raw_data:
        if cls:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        if this_dist > last_dist:
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
        last_dist = this_dist
    tpr.append(1)
    fpr.append(1)

    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name))

    print("{}: {:.4f}; loss: {:.4f}".format(fid, metrics.auc(fpr, tpr), loss))

fig.update_yaxes(range=(0, 1))
fig.update_xaxes(range=(0, 1))
fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
fig.show()
fig.update_layout(width=650, height=400)
