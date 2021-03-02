"""Auxiliary file for splitting the identities to trajectories based on the camera"""


import argparse
import sqlalchemy.orm
import typing
import tqdm

from . import custom_argparse
from . import database_connection as connection
from . import utils


def main(argument_string: typing.Optional[str] = None):
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)

    argument_parser.add_argument('-c', '--camera-setup', type=int, required=True,
                                 help='ID of camera setup to transfer identities from')
    argument_parser.add_argument('-i', '--identity-model', type=int, required=True,
                                 help='ID of identity model to use for generation of trajectories')

    parsed_arguments = argument_parser.parse_args(argument_string)
    logger = utils.get_logger('Delegate identities', parsed_arguments.log)

    logger.debug("Querying identity model")
    try:
        identity_model = (
            connection.session
            .query(connection.IdentityModel)
            .filter_by(id=parsed_arguments.identity_model).one()
        )
    except sqlalchemy.orm.exc.NoResultFound:
        raise RuntimeError("No identity model with given ID found")

    logger.debug("Searching for matching trajectory model")
    try:
        trajectory_model = (
            connection.session
            .query(connection.TrajectoryModel)
            .filter_by(
                generator_name=identity_model.generator_name,
                options=identity_model.options,
                feature_type_id=identity_model.feature_type_id,
            )
            .one()
        )
    except sqlalchemy.orm.exc.NoResultFound:
        raise RuntimeError("No equivalent trajectory model found")

    logger.debug("Using trajectory model {}".format(trajectory_model.id))

    identities = (
        connection.session
        .query(connection.Identity)
        .filter_by(identity_model_id=parsed_arguments.identity_model, camera_setup_id=parsed_arguments.camera_setup)
        .options(
            sqlalchemy.orm
            .joinedload(connection.Identity.identity_detections)
            .joinedload(connection.IdentityDetection.detection)
            .joinedload(connection.Detection.frame),
            # sqlalchemy.orm.defaultload(connection.IdentityDetection.detection).defer(connection.Detection.crop)
        )
    )

    def generate_new_trajectory(cam_id: int) -> int:
        trajectory = connection.Trajectory(
            trajectory_model_id=trajectory_model.id,
            camera_id=cam_id,
        )
        connection.session.add(trajectory)
        connection.session.flush()
        return trajectory.id

    logger.debug("Querying identities")
    for identity in tqdm.tqdm(identities, desc="Processing identities", total=identities.count()):
        trajectories_by_camera = utils.ReactiveDefaultDict(generate_new_trajectory)
        for identity_detection in identity.identity_detections:
            camera_id = identity_detection.detection.frame.camera_id
            trajectory_id = trajectories_by_camera[camera_id]
            connection.session.add(connection.TrajectoryDetection(
                trajectory_id=trajectory_id,
                detection_id=identity_detection.detection_id,
            ))
    logger.debug("Committing results into database")
    connection.session.commit()
    logger.info("Successfully completed")


if __name__ == '__main__':
    main()
