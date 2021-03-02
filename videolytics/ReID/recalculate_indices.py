"""Script to recompute sequence_number and timestamp of the frames based on the primary key"""


import argparse
import datetime
import tqdm
from . import database_connection as connection
from . import utils
from . import custom_argparse


def main():
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)

    argument_parser.add_argument('-c', '--camera', type=int, required=True,
                                 help="ID of camera to recalculate indices in")
    argument_parser.add_argument('-d', '--start-date', '--date', default="2020-01-01 09:00:00",
                                 help="Datetime of first frame in YYYY-MM-DD HH:MM:SS")
    argument_parser.add_argument('-f', '--fps', default=None, type=float,
                                 help="Frame-rate of the source video; in unspecified, loaded from the database")

    parsed_arguments = argument_parser.parse_args()
    logger = utils.get_logger("Re-calculator", parsed_arguments.log)

    allowed_formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"]
    for format_ in allowed_formats:
        try:
            start_date = datetime.datetime.strptime(parsed_arguments.start_date, format_)
            break
        except ValueError:
            pass
    else:
        argument_parser.error("Start date has invalid format")

    camera = connection.session.query(connection.Camera).filter_by(id=parsed_arguments.camera).one_or_none()
    if camera is None:
        argument_parser.error("Camera with given id does not exists")
        return
    fps = parsed_arguments.fps if parsed_arguments.fps is not None else camera.fps

    logger.debug("Fetching frames")
    frames = connection.session.query(connection.Frame).filter_by(camera_id=camera.id).order_by(connection.Frame.id)

    objects_for_update = list()
    for sequence_number, frame in tqdm.tqdm(enumerate(frames), total=frames.count(), desc="Processing frames"):
        seconds_from_start = sequence_number / fps
        frame.sequence_number = sequence_number
        frame.timestamp = start_date + datetime.timedelta(seconds=seconds_from_start)
        objects_for_update.append(frame)

    logger.debug("Committing data into database")
    connection.session.bulk_save_objects(objects_for_update)
    connection.session.commit()
    logger.info("Data successfully updated")


if __name__ == '__main__':
    main()
