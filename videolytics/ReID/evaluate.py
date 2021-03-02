"""Additional evaluation scripts not used in the thesis"""


import argparse
import sqlalchemy.orm
import tqdm
import typing
from . import database_connection as connection
from . import custom_argparse
from . import utils


def main():
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)

    argument_parser.add_argument('-i', '-t', '--collection', '--trajectory', '--identity', type=int, required=True,
                                 help="ID of collection (trajectory or identity) model to evaluate")
    argument_parser.add_argument('-g', '--golden', type=int, required=True,
                                 help="ID of golden collection")
    argument_parser.add_argument('-r', '--reidentification', action='store_true',
                                 help="Evaluate identities instead of trajectories")
    argument_parser.add_argument('-c', '--cameras', '--camera', '--camera_setup', type=int, required=True,
                                 help="ID of camera (or camera setup in case of reidentification) to evaluate "
                                      "the model in")
    argument_parser.add_argument('-p', '--page-size', type=int, default=5000,
                                 help='Number of detection requested from database at one time. Setting this to low '
                                      'number decreases local RAM usage but increases the number of database requests. '
                                      'To disable paging altogether, set this to 0')

    parsed_arguments = argument_parser.parse_args()
    evaluate(**vars(parsed_arguments))


def evaluate(collection: int, golden: int, reidentification: bool, cameras: int, page_size: int = 5000,
             log: typing.Optional[str] = None):
    logger = utils.get_logger("Evaluate", log)

    conf_matrix: utils.TwoDimensionalDict[int, int, int] = utils.TwoDimensionalDict()

    logger.debug("Establishing database query")
    if reidentification:
        GoldenIdentityDetection = sqlalchemy.orm.aliased(connection.IdentityDetection)
        GoldenIdentity = sqlalchemy.orm.aliased(connection.Identity)
        QueryIdentityDetection = sqlalchemy.orm.aliased(connection.IdentityDetection)
        QueryIdentity = sqlalchemy.orm.aliased(connection.Identity)
        collection_name = 'identity'

        query = (
            connection.session
            .query(GoldenIdentityDetection, QueryIdentityDetection)
            .join(GoldenIdentity, GoldenIdentityDetection.identity_id == GoldenIdentity.id)
            .outerjoin(QueryIdentityDetection, QueryIdentityDetection.detection_id == GoldenIdentityDetection.detection_id)
            .join(QueryIdentity, QueryIdentityDetection.identity_id == QueryIdentity.id)
            .filter(sqlalchemy.and_(
                GoldenIdentity.camera_setup_id == cameras,
                QueryIdentity.camera_setup_id == cameras,
                GoldenIdentity.identity_model_id == golden,
                QueryIdentity.identity_model_id == collection,
            ))
            .order_by(GoldenIdentityDetection.id, QueryIdentityDetection.id)
        )
    else:
        GoldenTrajectoryDetection = sqlalchemy.orm.aliased(connection.TrajectoryDetection, name='golden_trajectory_detection')
        GoldenTrajectory = sqlalchemy.orm.aliased(connection.Trajectory, name='golden_trajectory')
        QueryTrajectoryDetection = sqlalchemy.orm.aliased(connection.TrajectoryDetection, name='query_trajectory_detection')
        QueryTrajectory = sqlalchemy.orm.aliased(connection.Trajectory, name='query_trajectory')
        collection_name = 'trajectory'

        query = (
            connection.session
            .query(GoldenTrajectoryDetection, QueryTrajectoryDetection)
            .join(GoldenTrajectory, GoldenTrajectoryDetection.trajectory_id == GoldenTrajectory.id)
            .outerjoin(QueryTrajectoryDetection, QueryTrajectoryDetection.detection_id == GoldenTrajectoryDetection.detection_id)
            .join(QueryTrajectory, QueryTrajectoryDetection.trajectory_id == QueryTrajectory.id)
            .filter(sqlalchemy.and_(
                GoldenTrajectory.camera_id == cameras,
                QueryTrajectory.camera_id == cameras,
                GoldenTrajectory.trajectory_model_id == golden,
                QueryTrajectory.trajectory_model_id == collection,
            ))
            .order_by(GoldenTrajectoryDetection.id, QueryTrajectoryDetection.id)
        )
    collection_id_name = collection_name + '_id'

    total = query.count()
    if page_size > 0:
        query = utils.smooth_paged_query(query, page_size)
    iterator = tqdm.tqdm(query, total=total, desc="Computing confusion matrix")
    for golden_collection_detection, query_collection_detection in iterator:
        key = getattr(golden_collection_detection, collection_id_name), getattr(query_collection_detection, collection_id_name)
        if key not in conf_matrix.data:
            conf_matrix[key] = 1
        else:
            conf_matrix[key] += 1

    logger.debug("Computing hits per collection")
    golden_hits = {golden_collection_id: sum(hits_per_query.values())
                   for golden_collection_id, hits_per_query in conf_matrix.standard_ordered_data.items()}
    query_hits = {query_collection_id: sum(hits_per_golden.values())
                  for query_collection_id, hits_per_golden in conf_matrix.reverse_ordered_data.items()}

    logger.debug("Computing f1 scores")
    f1_scores: utils.TwoDimensionalDict[int, int, float] = utils.TwoDimensionalDict()
    iterator = tqdm.tqdm(conf_matrix.data.keys(), desc="Computing f1 scores")
    for key in iterator:
        golden_collection_id, query_collection_id = key
        f1_scores[key] = 2 * conf_matrix[key] / (golden_hits[golden_collection_id] + query_hits[query_collection_id])

    logger.debug("Evaluating best collection matching and selecting related f1 scores")
    best_f1_scores = {golden_collection_id: max(f1_scores_per_query.values())
                      for golden_collection_id, f1_scores_per_query in f1_scores.standard_ordered_data.items()}
    average_f1_score = sum(best_f1_scores.values()) / len(best_f1_scores)

    print("Computed f1 score of collection model {}: {}".format(collection, average_f1_score))


if __name__ == '__main__':
    main()