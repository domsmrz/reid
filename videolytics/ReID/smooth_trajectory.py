"""Smoothing of the trajectories, not described in the thesis"""


import argparse
import collections
import datetime
import typing
import sqlalchemy.orm
import tqdm
from . import utils
from . import custom_argparse
from . import database_connection as connection


def main():
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)
    argument_parser.add_argument('-c', '--camera', type=int, required=True,
                                 help="Smooth trajectories on the camera with this id")
    argument_parser.add_argument('-t', '--trajectory-model', type=int, required=True,
                                 help="Trajectory model to smooth")
    argument_parser.add_argument('-p', '--page-size', type=int, default=5000,
                                 help='Number of detection requested from database at one time. Setting this to low'
                                      'number decreases local RAM usage but increases the number of database requests.'
                                      'To disable paging altogether, set this to 0')
    parsed_arguments = argument_parser.parse_args()
    smooth(**vars(parsed_arguments))


def smooth(camera: int, trajectory_model: int, page_size: int = 5000, log: typing.Optional[str] = None):
    logger = utils.get_logger('Smooth Trajectory', log)

    detection_query = (
        connection.session
        .query(connection.Detection, connection.Trajectory)
        .select_from(connection.Detection)
        .join(connection.TrajectoryDetection)
        .join(connection.Trajectory)
        .join(connection.Frame)
        .options(
            sqlalchemy.orm.defer(connection.Detection.crop),
            sqlalchemy.orm.eagerload(connection.Detection.frame),
            sqlalchemy.orm.contains_eager(connection.Detection.frame),
        )
        .filter(connection.Trajectory.trajectory_model_id == trajectory_model)
        .filter(connection.Frame.camera_id == camera)
        .order_by(connection.Frame.id, connection.Detection.id)
    )

    frame_query = (
        connection.session
        .query(connection.Frame)
        .filter(connection.Frame.camera_id == camera)
        .order_by(connection.Frame.id)
    )

    frame_count = frame_query.count()
    logger.debug("Total detections to smooth out: {}".format(detection_query.count()))

    if page_size > 0:
        frame_query = utils.smooth_paged_query(frame_query, page_size)
        detection_query = utils.smooth_paged_query(detection_query, page_size)

    frame_iterator = tqdm.tqdm(frame_query, total=frame_count, desc="Processing frames")
    detection_iterator = utils.PeekableIterator(detection_query)

    trajectory_memory: typing.DefaultDict[connection.Trajectory, typing.Deque[connection.Detection]] =\
        collections.defaultdict(collections.deque)
    smooth_points: typing.List[connection.TrajectorySmooth] = list()

    connecting_timedelta = datetime.timedelta(milliseconds=500)
    for frame in frame_iterator:
        next_detection = detection_iterator.peek()
        while (
                next_detection is not detection_iterator.END
                and frame.timestamp + connecting_timedelta >= next_detection.Detection.frame.timestamp
        ):
            detection, trajectory = next(detection_iterator)
            trajectory_memory[trajectory].append(detection)
            next_detection = detection_iterator.peek()

        active_trajectories = list(trajectory_memory.keys())
        for trajectory in active_trajectories:
            memory = trajectory_memory[trajectory]
            while memory[0].frame.timestamp + connecting_timedelta < frame.timestamp:
                memory.popleft()

            if memory[-1].frame.sequence_number < frame.sequence_number:
                del trajectory_memory[trajectory]
                continue
            if memory[0].frame.sequence_number > frame.sequence_number:
                continue

            past = 0
            future = 0
            for detection in memory:
                if detection.frame.sequence_number <= frame.sequence_number:
                    past += 1
                if detection.frame.sequence_number >= frame.sequence_number:
                    future += 1
            total = past + future
            attributes = {
                'x': None,
                'y': None,
            }
            for attribute in attributes.keys():
                weight_sum = 0
                memory_detection_iterator = iter(sorted(memory, key=lambda det: getattr(det, 'center_' + attribute)))
                while weight_sum < total / 2:
                    detection = next(memory_detection_iterator)
                    if detection.frame.sequence_number <= frame.sequence_number:
                        weight_sum += future
                    if detection.frame.sequence_number >= frame.sequence_number:
                        weight_sum += past
                attributes[attribute] = getattr(detection, 'center_' + attribute)
            smooth_points.append(connection.TrajectorySmooth(
                trajectory_id=trajectory.id,
                frame_id=frame.id,
                **attributes,
            ))

    logger.debug("Smooth points constructed; committing into database")
    connection.session.bulk_save_objects(smooth_points)
    connection.session.commit()
    logger.info("Points committed; total number of smoothed out points: {}".format(len(smooth_points)))


if __name__ == '__main__':
    main()