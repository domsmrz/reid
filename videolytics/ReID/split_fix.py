"""Auxiliary file to split the existing trajectories based on the golden trajectory, used for develompent only"""


import argparse
import logging
import typing
import sqlalchemy.orm
import tqdm
import itertools
from . import database_connection as connection


def get_trajectory_model(trajectory_model_id: int) -> connection.TrajectoryModel:
    return (
        connection.session
        .query(connection.TrajectoryModel)
        .options(
            sqlalchemy.orm
            .joinedload(connection.TrajectoryModel.trajectories)
            .joinedload(connection.Trajectory.trajectory_detections)
        )
        .filter(connection.TrajectoryModel.id == trajectory_model_id)
        .one()
    )


def main(argument_string: typing.Optional[str] = None):
    logger = logging.getLogger('Evaluator')
    logger_stream_handler = logging.StreamHandler()
    logger_stream_handler.setLevel(0)
    logger_formatter = logging.Formatter('[{asctime}] {name} [{levelname}]: {message}', style='{')
    logger_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_stream_handler)
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-g', '--golden', type=int, required=True,
                                 help="Golden trajectory ID")
    argument_parser.add_argument('-t', '--trajectory', type=int, required=True,
                                 help="ID of trajectory to split-fix")
    argument_parser.add_argument('-c', '--camera', type=int, required=True,
                                 help="ID of camera to perform split-fix on")
    argument_parser.add_argument('--log', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str,
                                 help='Threshold for logger level')

    parsed_arguments = argument_parser.parse_args(argument_string)
    if parsed_arguments.log is not None:
        logger.setLevel(logging.getLevelName(parsed_arguments.log))

    logger.debug("Fetching golden trajectories")
    golden_trajectory_model = get_trajectory_model(parsed_arguments.golden)
    logger.debug("Fetching query trajectories")
    query_trajectory_model = get_trajectory_model(parsed_arguments.trajectory)

    golden_detections = [
        {trajectory_detection.detection_id for trajectory_detection in trajectory.trajectory_detections}
        for trajectory in golden_trajectory_model.trajectories
        if trajectory.camera_id == parsed_arguments.camera
    ]
    query_detections = [
        {trajectory_detection.detection_id for trajectory_detection in trajectory.trajectory_detections}
        for trajectory in query_trajectory_model.trajectories
        if trajectory.camera_id == parsed_arguments.camera
    ]

    fixed_model = connection.TrajectoryModel(
        description="Split-fixed: {}".format(query_trajectory_model.description),
        feature_type=query_trajectory_model.feature_type,
    )
    connection.session.add(fixed_model)

    iterator = tqdm.tqdm(itertools.product(golden_detections, query_detections), desc="Splitting trajectories",
                         total=len(golden_detections) * len(query_detections))
    for golden_detection_set, query_detection_set in iterator:
        common_detection_ids = golden_detection_set & query_detection_set
        if not common_detection_ids:
            continue
        trajectory = connection.Trajectory(trajectory_model=fixed_model, camera_id=parsed_arguments.camera)
        connection.session.add(trajectory)
        for detection_id in common_detection_ids:
            connection.session.add(connection.TrajectoryDetection(trajectory=trajectory, detection_id=detection_id))

    logger.debug("Committing results into database")
    connection.session.commit()
    logger.info("Split-fixed trajectory model committed with ID {}".format(fixed_model.id))


if __name__ == '__main__':
    main()
