"""Helper script to split the view of the camera to two artificial cameras, used for development"""


import argparse
import collections
import logging
import os
import sqlalchemy.orm.exc
import tqdm
import typing

from . import custom_argparse
from . import database_connection as connection
from . import utils


def split_camera(original_camera_id: int, widths: typing.List[int], heights: typing.List[int], xs: typing.List[int],
                 ys: typing.List[int], logger: logging.Logger, report: typing.Optional[typing.IO] = None,
                 trajectory_model_id: typing.Optional[int] = None):

    n_output_cameras = min(map(len, [xs, ys, widths, heights]))
    if n_output_cameras != max(map(len, [xs, ys, widths, heights])):
        raise RuntimeError("Invalid number of parameters")

    left_boundaries = xs
    right_boundaries = [x + width for x, width in zip(xs, widths)]
    top_boundaries = ys
    bottom_boundaries = [y + height for y, height in zip(ys, heights)]

    write_report = report.write if report is not None else lambda _: None

    logger.debug("Fetching data")
    try:
        original_camera = (
            connection.session
            .query(connection.Camera)
            .options(
                sqlalchemy.orm
                .joinedload(connection.Camera.frames)
                .joinedload(connection.Frame.detections)
            )
            .filter(connection.Camera.id == original_camera_id)
            .one()
        )
    except sqlalchemy.orm.exc.NoResultFound:
        raise RuntimeError("No camera with given ID found")
    logger.debug("Data fetched")

    new_cameras = [connection.Camera(
        name="{}[{},{}-{},{}]".format(original_camera.name, x, y, x+width, y+height),
        fps=original_camera.fps,
        h=height,
        w=width,
        ip="0.0.0.0",
        port=0,
        codec="none",
    ) for x, y, width, height in zip(xs, ys, widths, heights)]

    connection.session.add_all(new_cameras)
    connection.session.flush()
    for new_camera in new_cameras:
        write_report("camera {} {}\n".format(original_camera.id, new_camera.id))

    camera_setup = connection.CameraSetup(
        description="Artificial camera setup by splitting camera {}".format(original_camera_id)
    )
    connection.session.add(camera_setup)
    connection.session.flush()

    camera_setup_assignments = [connection.CameraSetupAssignment(
        camera_setup_id=camera_setup.id,
        camera_id=new_camera.id
    ) for new_camera in new_cameras]
    connection.session.bulk_save_objects(camera_setup_assignments)
    connection.session.flush()

    original_detections_ids_to_new_detection_ids = dict()

    for original_frame in tqdm.tqdm(original_camera.frames, desc="Processing frames"):
        new_frames = [connection.Frame(
            camera=new_camera,
            timestamp=original_frame.timestamp,
            sequence_number=original_frame.sequence_number,
        ) for new_camera in new_cameras]
        connection.session.add_all(new_frames)
        connection.session.flush()
        for new_camera, new_frame in zip(new_cameras, new_frames):
            write_report("frame {} {} # camera: {}\n".format(original_frame.id, new_frame.id, new_camera.id))

        for original_detection in original_frame.detections:
            for left_boundary, right_boundary, top_boundary, bottom_boundary, x, y, new_frame, new_camera\
                    in zip(left_boundaries, right_boundaries, top_boundaries, bottom_boundaries, xs, ys, new_frames,
                           new_cameras):
                if (
                    original_detection.left < left_boundary
                    or original_detection.right >= right_boundary
                    or original_detection.top < top_boundary
                    or original_detection.bottom >= bottom_boundary
                ):
                    continue
                new_detection = connection.Detection(
                    frame=new_frame,
                    class_=original_detection.class_,
                    left=original_detection.left - x,
                    top=original_detection.top - y,
                    right=original_detection.right - x,
                    bottom=original_detection.bottom - y,
                    crop=original_detection.crop,
                    conf=original_detection.conf,
                )
                connection.session.add(new_detection)
                connection.session.flush()
                original_detections_ids_to_new_detection_ids[original_detection.id] = new_detection.id
                write_report("detection {} {} # camera: \n".format(original_detection.id, new_detection.id,
                                                                   new_camera.id))

    if trajectory_model_id is not None:
        trajectory_detections = (
            connection.session
            .query(connection.TrajectoryDetection)
            .join(connection.Trajectory)
            .filter(connection.Trajectory.trajectory_model_id == trajectory_model_id)
        )

        trajectory_model = connection.session.query(connection.TrajectoryModel).filter_by(id=trajectory_model_id).one()
        identity_model_parameters = dict(
            generator_name=trajectory_model.generator_name,
            options=trajectory_model.options,
            feature_type_id=trajectory_model.feature_type_id,
            trajectory_model_id=trajectory_model.id,
        )
        identity_model = (
            connection.session
            .query(connection.IdentityModel)
            .filter_by(**identity_model_parameters)
            .one_or_none()
        )
        if identity_model is None:
            identity_model = connection.IdentityModel(**identity_model_parameters)
            connection.session.add(identity_model)
            connection.session.flush()

        def submit_identity(original_trajectory_id: int) -> int:
            identity = connection.Identity(
                identity_model_id=identity_model.id,
                camera_setup_id=camera_setup.id,
            )
            connection.session.add(identity)
            connection.session.flush()
            write_report("trajectory {} {}\n".format(original_trajectory_id, identity.id))
            return identity.id

        original_trajectory_ids_to_new_identity_ids: utils.ReactiveDefaultDict[int, int] =\
            utils.ReactiveDefaultDict(submit_identity)
        new_identity_detections = list()
        for trajectory_detection in tqdm.tqdm(trajectory_detections, desc="Processing trajectory detections",
                                              total=trajectory_detections.count()):
            if trajectory_detection.detection_id not in original_detections_ids_to_new_detection_ids:
                continue
            new_identity_id = original_trajectory_ids_to_new_identity_ids[trajectory_detection.trajectory_id]
            new_detection_id = original_detections_ids_to_new_detection_ids[trajectory_detection.detection_id]
            new_identity_detection = connection.IdentityDetection(
                identity_id=new_identity_id,
                detection_id=new_detection_id,
            )
            new_identity_detections.append(new_identity_detection)
        connection.session.bulk_save_objects(new_identity_detections)
        connection.session.flush()

    answer = ''
    while answer.lower() not in ['n', 'no']:
        answer = input("Commit? ")
        if answer.lower() in ['y', 'yes']:
            connection.session.commit()
            logger.info("Completed, new camera setup added with id {}".format(camera_setup.id))
            break
    else:
        logger.info("Committing aborted")


def main():
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)
    argument_parser.add_argument('-c', '--camera', required=True, type=int,
                                 help="ID of camera to split")
    argument_parser.add_argument('--width', required=True, type=int, nargs="+",
                                 help="Width of the crop")
    argument_parser.add_argument('--height', required=True, type=int, nargs="+",
                                 help="Height of the crop")
    argument_parser.add_argument('-x', '--x', '--left', required=True, type=int, nargs="+",
                                 help="X-coordinate of the top-left corner of the crop")
    argument_parser.add_argument('-y', '--y', '--top', required=True, type=int, nargs="+",
                                 help="Y-coordinate of the top-left corner of the crop")
    argument_parser.add_argument('-r', '--report', type=str,
                                 help="Name of file to dump the report into")
    argument_parser.add_argument('-t', '--trajectory-model', type=int,
                                 help="ID of trajectory model to export into newly generated camera")
    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.report is not None and os.path.isfile(parsed_arguments.report):
        argument_parser.error("Chosen report file {!r} exists, please choose different location".format(
            parsed_arguments.report))

    parameters_to_normalize = ['width', 'height', 'x', 'y']
    normalized_parameters = dict()
    n_output_cameras = max(len(getattr(parsed_arguments, parameter)) for parameter in parameters_to_normalize)
    for parameter in parameters_to_normalize:
        value = getattr(parsed_arguments, parameter)
        if len(value) == 1:
            normalized_parameters[parameter] = n_output_cameras * value
        elif len(value) == n_output_cameras:
            normalized_parameters[parameter] = value
        else:
            argument_parser.error("Number of parameters in option {} does not match expected number of output cameras"
                                  "".format(parameter, n_output_cameras))

    logger = utils.get_logger('Annotator', parsed_arguments.log)

    with open(parsed_arguments.report, "w") as f:
        split_camera(
            original_camera_id=parsed_arguments.camera,
            widths=normalized_parameters['width'],
            heights=normalized_parameters['height'],
            xs=normalized_parameters['x'],
            ys=normalized_parameters['y'],
            report=f,
            trajectory_model_id=parsed_arguments.trajectory_model,
            logger=logger,
        )


if __name__ == '__main__':
    main()