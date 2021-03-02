"""Auxiliary script for migrating from old table structure for trajectories to new one"""


import argparse
import logging
import typing
from . import database_connection as connection
import tqdm


argument_parser = argparse.ArgumentParser()
parsed_arguments = None
logger = logging.getLogger('TrajectoryMigrator')
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(0)
logger_formatter = logging.Formatter('[{asctime}] {name} [{levelname}]: {message}', style='{')
logger_stream_handler.setFormatter(logger_formatter)
logger.addHandler(logger_stream_handler)


def main(argument_string: typing.Optional[str] = None):
    global parsed_arguments
    argument_parser.add_argument('--log', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str,
                                 help='Threshold for logger level')
    argument_parser.add_argument('-m', '--model', type=int,
                                 help='Trajectory Model ID to migrate to new format')

    parsed_arguments = argument_parser.parse_args(argument_string)
    if parsed_arguments.log is not None:
        logger.setLevel(logging.getLevelName(parsed_arguments.log))

    logger.debug("Fetching old model from the database")
    old_model = connection.session.query(connection.OldTrajectoryModel).filter_by(id=parsed_arguments.model).one()
    old_detections = connection.session.query(connection.OldTrajectoryDetection).filter_by(model_id=parsed_arguments.model)

    logger.debug("Creating new model")
    new_model = connection.TrajectoryModel(
        generator_name='manual-annotation',
        options='',
        description="Migrated: {}".format(old_model.description),
        feature_type_id=None,
    )
    connection.session.add(new_model)

    trajectory_mapping = dict()

    iterator = tqdm.tqdm(old_detections, desc="Processing detections", total=old_detections.count())
    for old_trajectory_detection in iterator:
        if old_trajectory_detection.trajectory_id not in trajectory_mapping:
            detection = connection.session.query(connection.Detection).filter_by(id=old_trajectory_detection.detection_id).one()
            frame = detection.frame
            camera = frame.camera
            new_trajectory = connection.Trajectory(
                trajectory_model=new_model,
                camera=camera,
            )
            connection.session.add(new_trajectory)
            trajectory_mapping[old_trajectory_detection.trajectory_id] = new_trajectory
            
        else:
            new_trajectory = trajectory_mapping[old_trajectory_detection.trajectory_id]

        new_trajectory_detection = connection.TrajectoryDetection(
            trajectory=new_trajectory,
            detection_id=old_trajectory_detection.detection_id,
        )
        connection.session.add(new_trajectory_detection)

    logger.debug("Data prepared for commit into the database")
    connection.session.commit()
    logger.info("Data successfully generated and committed into the database with model ID {}".format(new_model.id))


if __name__ == '__main__':
    main()
