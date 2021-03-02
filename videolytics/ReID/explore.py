"""Annotation tool

check the user guide for further description"""


import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageTk
import argparse
import collections
import datetime
import enum
import functools
import glob
import logging
import numpy as np
import os
import operator
import pickle
import shlex
import sqlalchemy
import sqlalchemy.orm
import string
import tkinter
import tqdm
import typing

from . import custom_argparse
from . import database_connection as connection
from . import utils


class SelectionType(enum.Enum):
    Left = 'left'
    Right = 'right'


class HighlightColor(enum.Enum):
    LEFT = 'red'
    RIGHT = 'green'
    AVAILABLE = 'blue'
    HIGHLIGHTED = 'cyan'
    UNAVAILABLE = 'yellow'


class SameTimeFrames(typing.NamedTuple):
    timestamp: datetime.datetime
    frames: typing.List[typing.Optional[connection.Frame]]


class AppFix:
    TEXT_HEIGHT = 25
    TEXT_WIDTH = 11
    SMALL_TEXT_HEIGHT = 16
    SMALL_TEXT_WIDTH = 8
    font: PIL.ImageFont.FreeTypeFont = PIL.ImageFont.truetype("Hack-Regular.ttf", 18)
    small_font: PIL.ImageFont.FreeTypeFont = PIL.ImageFont.truetype("Hack-Regular.ttf", 13)

    def __init__(self, root: tkinter.Tk, camera_ids: typing.List[int], file: typing.Optional[str] = None,
                 trajectory_model_id: typing.Optional[int] = None, identity_model_id: typing.Optional[int] = None,
                 log: typing.Optional[str] = None, preload: bool = False, scale: float = 1.0,
                 class_: typing.Optional[str] = None, start_time_str: typing.Optional[str] = None,
                 end_time_str: typing.Optional[str] = None, **_):
        self.logger = utils.get_logger('Explore', log)

        self.logger.debug("Loading camera")
        self.cameras: typing.List[connection.Camera] = (
            connection.session
            .query(connection.Camera)
            .filter(connection.Camera.id.in_(camera_ids))
            .order_by(connection.Camera.id)
            .all()
        )
        if len(self.cameras) != len(camera_ids):
            raise RuntimeError("Invalid camera id(s)")

        self.dimensions = [np.array([camera.w, camera.h], np.int32) for camera in self.cameras]
        self.stack_dimension: int = 0
        self.dimension_offsets = [np.zeros(2, np.int32)]
        for single_dimensions in self.dimensions:
            add = np.zeros(2, np.int32)
            add[self.stack_dimension] += single_dimensions[self.stack_dimension]
            self.dimension_offsets.append(self.dimension_offsets[-1] + add)

        self.shown_class = class_.upper() if isinstance(class_, str) else None
        self.root = root
        self.root.bind("<KeyPress>", self.key_handler)
        self.last_command_without_parameters: typing.Optional[typing.Callable[[], typing.Any]] = None

        self.enclosing_dimensions: typing.Tuple[int, ...] = tuple(
            (sum if i == self.stack_dimension else max)(single_dimension_vector)
            for i, single_dimension_vector in enumerate(zip(*self.dimensions))
        )

        self.scale: float = scale
        self.keys_since_quit: int = 1
        self.showing_single_collection: bool = False
        self.background_image: PIL.Image.Image = PIL.Image.new("RGB", self.enclosing_dimensions)
        self.draw: PIL.ImageDraw.ImageDraw = PIL.ImageDraw.Draw(self.background_image)
        tk_image = PIL.ImageTk.PhotoImage(self.background_image)

        self.canvas = tkinter.Canvas(self.root, width=self.enclosing_dimensions[0], height=self.enclosing_dimensions[1])
        self.canvas.pack(fill="x", expand=True)
        # self.label = tkinter.Label(self.root, image=tk_image)

        self.label = tkinter.Label(self.canvas, image=tk_image)
        self.label.image = tk_image
        self.label.bind("<Button>", self.button_handler)
        self.label.bind("<Double-Button>", self.double_button_handler)
        self.label.pack()

        self.canvas.create_window(0, 0, anchor="nw", window=self.label)

        self.basic_commands: typing.Dict[str, typing.Callable[[], typing.Optional[str]]] = dict()
        self.commands_without_parameters: typing.Dict[str, typing.Callable[[], typing.Optional[str]]] = dict()
        self.short_commands_without_parameters: typing.Dict[str, typing.Callable[[], typing.Optional[str]]] = dict()
        self.commands_with_parameters: typing.Dict[str, typing.Callable[[str], typing.Optional[str]]] = dict()
        self.setup_key_handler_signpost()
        self.active_command: typing.Optional[str] = None
        self.showing_ids: bool = False
        self.loaded_pairs: typing.Optional[typing.Iterator[typing.Tuple[int, int]]] = None
        self.loaded_commented_rows: typing.Optional[typing.List[str]] = None
        self.last_loaded_pair: typing.Optional[typing.Tuple[int, int]] = None

        self.show_unassigned: bool = True
        self.selected_collection_id_left: typing.Optional[int] = None
        self.selected_collection_id_right: typing.Optional[int] = None
        self.last_search: typing.Optional[typing.Tuple[int, int]] = None
        self.hidden_collection_ids: typing.Set[int] = set()
        self.frame_index: int = 0
        self.shown_detections: typing.List[connection.Detection] = list()
        self.highlighted_collection_ids: typing.Set[int] = set()

        self.logger.debug("Establishing frame queries")

        start_timestamp = (
            connection.session
            .query(sqlalchemy.func.min(connection.Frame.timestamp))
            .filter(connection.Frame.camera_id.in_(camera_ids))
            .scalar()
        )

        self.preload: bool = preload
        frames_queries = list()

        for camera_id in camera_ids:
            query = (
                connection.session
                .query(connection.Frame)
                .filter(connection.Frame.camera_id == camera_id)
                .order_by(connection.Frame.timestamp)
            )

            if preload:
                query = query.outerjoin(connection.Detection)

                if self.shown_class is not None:
                    query = query.filter(sqlalchemy.or_(
                        connection.Detection.class_ == self.shown_class,
                        connection.Detection.id.is_(None),
                    ))
                query = query.options(sqlalchemy.orm.contains_eager(connection.Frame.detections))

            if start_time_str is not None or end_time_str is not None:
                self.logger.debug("Applying time constraints")
                if start_time_str is not None:
                    start_time = datetime.datetime.strptime(start_time_str, "%H:%M:%S")
                    start_time_delta = datetime.timedelta(hours=start_time.hour, minutes=start_time.minute,
                                                          seconds=start_time.second)
                    query = query.filter(connection.Frame.timestamp >= start_timestamp + start_time_delta)
                if end_time_str is not None:
                    end_time = datetime.datetime.strptime(end_time_str, "%H:%M:%S")
                    end_time_delta = datetime.timedelta(hours=end_time.hour, minutes=end_time.minute,
                                                        seconds=end_time.second)
                    query = query.filter(connection.Frame.timestamp <= start_timestamp + end_time_delta)

            frames_queries.append(query)

        self.logger.debug("Loading frames and detections")
        cameras_frames_loaded = [query.all() for query in frames_queries]
        cameras_frames = [utils.PeekableIterator(data) for data in cameras_frames_loaded]

        self.detections_by_id = None
        if preload:
            self.detections_by_id = {
                detection.id: detection
                for frame_data in cameras_frames_loaded
                for frame in frame_data
                for detection in frame.detections
            }

        self.frames: typing.List[SameTimeFrames] = list()
        self.logger.debug("Initial processing of frames")
        while any(single_camera_frames.peek() != utils.PeekableIterator.END for single_camera_frames in cameras_frames):
            minimal_timestamp = min(
                single_camera_frames.peek().timestamp
                for single_camera_frames in cameras_frames
                if single_camera_frames.peek() is not utils.PeekableIterator.END
            )
            self.frames.append(SameTimeFrames(
                timestamp=minimal_timestamp,
                frames=[(
                    next(single_camera_frames) if (
                            single_camera_frames.peek() is not utils.PeekableIterator.END
                            and single_camera_frames.peek().timestamp == minimal_timestamp
                    ) else None
                ) for single_camera_frames in cameras_frames],
            ))

        self.collection_id_by_detection_id: typing.Dict[int, int]
        self.detection_ids_by_collection_id: typing.Dict[int, typing.Set[int]]
        self.camera_setup: typing.Optional[connection.CameraSetup] = None

        self.logger.debug("Loading collection-detection data")
        if file is not None:
            with open(file, 'rb') as f:
                pickled = pickle.load(f)
            for key, value in pickled.items():
                setattr(self, key, value)

        elif camera_ids is not None:
            if trajectory_model_id is not None:
                collection_id_str = 'trajectory_id'

                collection_detections = (
                    connection.session
                    .query(connection.TrajectoryDetection)
                    .join(connection.Trajectory)
                    .filter(connection.Trajectory.camera_id == camera_ids[0])
                    .filter(connection.Trajectory.trajectory_model_id == trajectory_model_id)
                )
            elif identity_model_id is not None:
                collection_id_str = 'identity_id'
                camera_ids_set = set(camera_ids)

                for camera_setup in connection.session.query(connection.CameraSetup):
                    if set(assignment.camera_id for assignment in camera_setup.camera_setup_assignments) == camera_ids_set:
                        break
                else:
                    raise RuntimeError("Given camera setup not found")

                self.camera_setup = camera_setup
                collection_detections = (
                    connection.session
                    .query(connection.IdentityDetection)
                    .join(connection.Identity)
                    .filter(connection.Identity.camera_setup_id == camera_setup.id)
                    .filter(connection.Identity.identity_model_id == identity_model_id)
                )
            else:
                RuntimeError("Invalid arguments")
                collection_detections = None
                collection_id_str = None

            self.collection_id_by_detection_id = {
                collection_detection.detection_id: getattr(collection_detection, collection_id_str)
                for collection_detection in collection_detections
            }

            self.detection_ids_by_collection_id = dict()
            for detection_id, collection_id in self.collection_id_by_detection_id.items():
                self.detection_ids_by_collection_id.setdefault(collection_id, set()).add(detection_id)

        else:
            RuntimeError("Invalid arguments for ApkFix")

        self.logger.debug("Finalizing initialization")
        self.last_artificial_collection_id = max(min(self.detection_ids_by_collection_id.keys()), 0) \
                                             if self.detection_ids_by_collection_id else 0
        self.logger.debug("Initialization complete")
        self.show_detections()

    @staticmethod
    def get_image_patch(width: int, height: int = TEXT_HEIGHT, color: str = 'black') -> PIL.Image.Image:
        return PIL.Image.new('RGB', (width, height), color)

    def display_text(self, display: PIL.Image.Image, text: str, position: typing.Tuple[int, int], color: str = "white",
                     background_color: typing.Optional[str] = None, small_font: bool = False) -> None:
        font = self.small_font if small_font else self.font
        text_width = self.SMALL_TEXT_WIDTH if small_font else self.TEXT_WIDTH
        text_height = self.SMALL_TEXT_HEIGHT if small_font else self.TEXT_HEIGHT
        if background_color is not None:
            patch = self.get_image_patch(width=text_width*len(text), height=text_height, color=background_color)
            display.paste(patch, position)
        draw = PIL.ImageDraw.Draw(display)
        draw.text(position, text, color, font=font)

    @property
    def current_frames(self) -> SameTimeFrames:
        return self.frames[self.frame_index]

    @property
    def start_time(self) -> datetime.datetime:
        return self.frames[0].timestamp

    @property
    def video_length(self) -> datetime.timedelta:
        return self.frames[-1].timestamp - self.start_time

    @staticmethod
    def format_timedelta(delta: datetime.timedelta) -> str:
        total_seconds = delta.seconds
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        total_hours = total_minutes // 60
        return "{:02}:{:02}:{:02}".format(total_hours, minutes, seconds)

    show_detections_report = collections.namedtuple('show_detections_report', ['shown_unselected'])

    def show_detections(self) -> show_detections_report:
        self.showing_single_collection = False
        self.shown_detections = list()
        shown_unselected = False
        display = self.background_image.copy()
        ids_to_display = list()

        for frame, frame_offset, camera in zip(self.current_frames.frames, self.dimension_offsets, self.cameras):
            if frame is None:
                continue
            detections_to_display = list()
            for detection in frame.detections:
                highlight_color = HighlightColor.AVAILABLE.value
                if self.shown_class is not None and detection.class_ != self.shown_class:
                    continue
                if (
                        detection.id not in self.collection_id_by_detection_id
                        or self.collection_id_by_detection_id[detection.id] in self.hidden_collection_ids
                ):
                    if not self.show_unassigned:
                        continue
                    highlight_color = HighlightColor.UNAVAILABLE.value
                else:
                    self.shown_detections.append(detection)
                collection_id = self.collection_id_by_detection_id.get(detection.id, None)

                if collection_id is not None:
                    if collection_id == self.selected_collection_id_left:
                        highlight_color = HighlightColor.LEFT.value
                    elif collection_id == self.selected_collection_id_right:
                        highlight_color = HighlightColor.RIGHT.value
                    elif collection_id in self.highlighted_collection_ids:
                        highlight_color = HighlightColor.HIGHLIGHTED.value
                shown_unselected = shown_unselected or highlight_color == HighlightColor.AVAILABLE.value
                detections_to_display.append((detection, highlight_color))

            enum_importance = [
                HighlightColor.LEFT,
                HighlightColor.RIGHT,
                HighlightColor.AVAILABLE,
                HighlightColor.HIGHLIGHTED,
                HighlightColor.UNAVAILABLE,
            ]
            color_importance = [x.value for x in enum_importance]

            for detection, highlight_color in sorted(detections_to_display, key=lambda x: color_importance.index(x[1]),
                                                     reverse=True):
                highlight = PIL.Image.new("RGB", tuple(reversed(np.int32(detection.dimensions * self.scale) + 10)),
                                          color=highlight_color)
                display.paste(highlight, tuple(np.array((detection.position + frame_offset) * self.scale, dtype=np.int32) - 5))
                if self.showing_ids:
                    position_y = detection.bottom + 5 if detection.bottom + 5 + self.SMALL_TEXT_HEIGHT <= camera.h\
                        else detection.top - self.SMALL_TEXT_HEIGHT - 5
                    ids_to_display.append(dict(
                        display=display,
                        text=str(detection.id),
                        position=tuple(np.int32((np.array([detection.left, position_y], np.int32)+frame_offset) * self.scale)),
                        color=highlight_color,
                        background_color='black',
                        small_font=True,
                    ))

                crop = PIL.Image.fromarray(detection.image)
                if self.scale != 1:
                    crop = crop.resize(tuple(reversed(np.int32(detection.dimensions * self.scale))), PIL.Image.ANTIALIAS)
                self.background_image.paste(crop, tuple(np.array((detection.position + frame_offset) * self.scale, np.int32)))
                display.paste(crop, tuple(np.array((detection.position + frame_offset) * self.scale, np.int32)))

            if self.showing_ids:
                ids_to_display.sort(key=lambda record: color_importance.index(record['color']), reverse=True)
                for id_to_display in ids_to_display:
                    self.display_text(**id_to_display)

            frame_progression = "{} / {}".format(self.frame_index, len(self.frames))
            self.display_text(display, frame_progression, (0, 0), background_color='black')
            current_time = self.format_timedelta(self.current_frames.timestamp - self.start_time)
            final_time = self.format_timedelta(self.video_length)
            time_progression = "{} / {}".format(current_time, final_time)
            self.display_text(display, time_progression, (0, self.TEXT_HEIGHT), background_color='black')

        max_camera_height = max(camera.h for camera in self.cameras)

        def count_detections(coll_id, op):
            return sum(
                op(self.detections_by_id[detection_id].frame.timestamp, self.current_frames.timestamp)
                for detection_id in self.detection_ids_by_collection_id[coll_id]
                if detection_id in self.detections_by_id
            )

        if self.active_command is None:
            shown_detection_ids = {detection.id for detection in self.shown_detections}
            if self.selected_collection_id_left is not None:
                left_detection_ids = ' '.join(
                    map(str, self.detection_ids_by_collection_id[self.selected_collection_id_left]
                        & shown_detection_ids)
                )
                extra_left_info_string = " {};{}".format(
                    count_detections(self.selected_collection_id_left, operator.lt),
                    count_detections(self.selected_collection_id_left, operator.gt)
                ) if self.preload else ''
                left_info_string = 'tr: {}; dt: {} ({}{})'.format(
                    self.selected_collection_id_left,
                    left_detection_ids,
                    len(self.detection_ids_by_collection_id[self.selected_collection_id_left]),
                    extra_left_info_string
                )
                self.display_text(display, left_info_string, (0, round(max_camera_height * self.scale) - self.TEXT_HEIGHT),
                                  color=HighlightColor.LEFT.value, background_color='black')
            if self.selected_collection_id_right is not None:
                right_detection_ids = ' '.join(
                    map(str, self.detection_ids_by_collection_id.get(self.selected_collection_id_right, set())
                        & shown_detection_ids)
                )
                extra_right_info_string = " {};{}".format(
                    count_detections(self.selected_collection_id_right, operator.lt),
                    count_detections(self.selected_collection_id_right, operator.gt),
                ) if self.preload else ''
                right_info_string = 'tr: {}; dt: {} ({}{})'.format(
                    self.selected_collection_id_right,
                    right_detection_ids,
                    len(self.detection_ids_by_collection_id[self.selected_collection_id_right]),
                    extra_right_info_string,
                )
                self.display_text(display, right_info_string, (0, round(max_camera_height * self.scale) - 2 * self.TEXT_HEIGHT),
                                  color=HighlightColor.RIGHT.value, background_color='black')
        else:
            self.display_text(display, ':' + self.active_command, (0, round(max_camera_height * self.scale) - self.TEXT_HEIGHT),
                              background_color='black')

        self.update_image(display)

        return self.show_detections_report(shown_unselected=shown_unselected)

    def update_image(self, image: PIL.Image.Image) -> None:
        tk_image = PIL.ImageTk.PhotoImage(image)
        self.label.configure(image=tk_image)
        self.label.image = tk_image

    def get_new_collection_id(self) -> int:
        self.last_artificial_collection_id -= 1
        return self.last_artificial_collection_id

    def seek(self, step: int) -> show_detections_report:
        self.frame_index += step
        if self.frame_index < 0:
            self.frame_index = 0
        if self.frame_index >= len(self.frames):
            self.frame_index = len(self.frames) - 1
        return self.show_detections()

    def seek_until_unselected(self, step: int) -> None:
        old_frame_index = None
        while True:
            show_detections_report = self.seek(step)
            if self.frame_index == old_frame_index:
                break
            old_frame_index = self.frame_index
            if show_detections_report.shown_unselected:
                break

    def clear_view(self):
        self.background_image = PIL.Image.new('RGB', self.enclosing_dimensions)
        self.show_detections()

    def unify(self) -> None:
        if self.selected_collection_id_left is None:
            return
        if self.selected_collection_id_right is None:
            return
        if self.selected_collection_id_left == self.selected_collection_id_right:
            return

        for detection_id in self.detection_ids_by_collection_id[self.selected_collection_id_right]:
            self.detection_ids_by_collection_id[self.selected_collection_id_left].add(detection_id)
            self.collection_id_by_detection_id[detection_id] = self.selected_collection_id_left
        del self.detection_ids_by_collection_id[self.selected_collection_id_right]


        self.selected_collection_id_right = None
        self.show_detections()

    def hide_collection(self) -> None:
        if self.selected_collection_id_left is None:
            return
        self.hidden_collection_ids.add(self.selected_collection_id_left)
        if self.selected_collection_id_right == self.selected_collection_id_left:
            self.selected_collection_id_right = None
        self.selected_collection_id_left = None
        self.show_detections()

    def swap_showing_unassigned(self) -> None:
        self.show_unassigned = not self.show_unassigned
        self.show_detections()

    def quit(self) -> None:
        if self.keys_since_quit <= 1:
            self.root.destroy()
        else:
            self.keys_since_quit = 0

    def save(self, filename: str, force: bool = False) -> str:
        if not force and os.path.exists(filename):
            return "File {} already exists".format(shlex.quote(filename))
        with open(filename, 'wb') as f:
            stored_parameters = ['detection_ids_by_collection_id', 'collection_id_by_detection_id']
            data = {key: getattr(self, key) for key in stored_parameters}
            pickle.dump(data, f)
        return "Data successfully saved in {}".format(shlex.quote(filename))

    def find(self, detection_id: int) -> bool:
        try:
            detection = (
                connection.session
                .query(connection.Detection)
                .filter(connection.Detection.id == detection_id)
                .one()
            )
        except sqlalchemy.orm.exc.NoResultFound:
            return False
        frame_id = detection.frame_id
        for frame_idx, frames_data in enumerate(self.frames):
            if any(frame is not None and frame.id == frame_id for frame in frames_data.frames):
                self.frame_index = frame_idx
                self.show_detections()
                return True
        else:
            return False

    def find_and_select(self, detection_id: int, select: SelectionType) -> bool:
        success = self.find(detection_id)
        if not success:
            return False
        if select is SelectionType.Left:
            self.selected_collection_id_left = self.collection_id_by_detection_id[detection_id]
        elif select is SelectionType.Right:
            self.selected_collection_id_right = self.collection_id_by_detection_id[detection_id]
        return True

    def find_command(self, parameters: str) -> str:
        detection_id_str, *rest = parameters.strip().split()
        detection_id = int(detection_id_str)
        if not rest:
            result = self.find(detection_id)
        else:
            try:
                right_detection_id = int(rest[0])
            except ValueError:
                selection_str = rest[0]
                if selection_str == 'l':
                    selection = SelectionType.Left
                elif selection_str == 'r':
                    selection = SelectionType.Right
                else:
                    return "Invalid selection {!r}".format(selection_str)
                result = self.find_and_select(detection_id, selection)
            else:
                self.selected_collection_id_right = self.collection_id_by_detection_id.get(right_detection_id)
                result = self.find_and_select(detection_id, SelectionType.Left)
                self.last_search = (detection_id, right_detection_id)
        if result:
            return "Detection {!r} found".format(detection_id)
        return "Detection {!r} not found".format(detection_id)

    def upload(self, description: str, loaded_only: bool = False) -> str:
        generator_name, options = description.split(' ', 1)

        if len(self.cameras) > 1:
            for camera_setup in connection.session.query(connection.CameraSetup):
                if set(assignment.camera_id for assignment in camera_setup.camera_setup_assignments):
                    break
            else:
                raise RuntimeError("Given camera setup not found")

            collection_model = connection.IdentityModel(
                generator_name=generator_name,
                options=options,
            )
            connection.session.add(collection_model)
            connection.session.flush()

            generate_collection = functools.partial(
                connection.Identity,
                identity_model_id=collection_model.id,
                camera_setup_id=camera_setup.id,
            )

            generate_detection = lambda collection_id, detection_id: connection.IdentityDetection(
                identity_id=collection_id,
                detection_id=detection_id,
            )
        else:
            collection_model = connection.TrajectoryModel(
                generator_name=generator_name,
                options=options,
            )
            connection.session.add(collection_model)
            connection.session.flush()

            generate_collection = functools.partial(
                connection.Trajectory,
                trajectory_model_id=collection_model.id,
                options=options,
            )

            generate_detection = lambda collection_id, detection_id: connection.TrajectoryDetection(
                trajectory_id=collection_id,
                detection_id=detection_id,
            )

        for detection_ids in self.detection_ids_by_collection_id.values():
            if loaded_only:
                detection_ids = [det_id for det_id in detection_ids if det_id in self.detections_by_id]
                if not detection_ids:
                    continue
            trajectory = generate_collection()
            connection.session.add(trajectory)
            connection.session.flush()
            detection_objects = [
                generate_detection(
                    collection_id=trajectory.id,
                    detection_id=detection_id
                ) for detection_id in detection_ids
            ]
            connection.session.bulk_save_objects(detection_objects)
            connection.session.flush()
        connection.session.commit()
        return "Data uploaded with id {}".format(collection_model.id)

    def set_frame(self, frame_index: int) -> None:
        if frame_index < 0:
            frame_index += len(self.frames)
        self.frame_index = frame_index
        self.show_detections()

    def find_selected(self, selection: SelectionType, forward_only: bool = True) -> None:
        collection_id = None
        if selection is SelectionType.Left:
            collection_id = self.selected_collection_id_left
        elif selection is SelectionType.Right:
            collection_id = self.selected_collection_id_right

        if collection_id is None:
            return
        if forward_only:
            frame_index_iterator = range(self.frame_index + 1, len(self.frames))
        else:
            frame_index_iterator = utils.intersperse_iterator(range(self.frame_index, len(self.frames)),
                                                              range(self.frame_index - 1, -1, -1))
        for frame_index in frame_index_iterator:
            frame_data = self.frames[frame_index]
            if any(
                    self.collection_id_by_detection_id.get(detection.id) == collection_id
                    for frame in frame_data.frames
                    if frame is not None
                    for detection in frame.detections
            ):
                self.frame_index = frame_index
                self.show_detections()
                return

    def transfer_detection(self, detection_id: typing.Optional[int] = None) -> None:
        if (
                self.selected_collection_id_left is None
                or (self.selected_collection_id_right is None and detection_id is None)
        ):
            return

        if detection_id is None:
            for detection in self.shown_detections:
                if self.collection_id_by_detection_id[detection.id] == self.selected_collection_id_right:
                    detection_id = detection.id
                    former_collection_id = self.selected_collection_id_right
                    break
            else:
                return
        else:
            try:
                former_collection_id = self.collection_id_by_detection_id[detection_id]
            except KeyError:
                return

        self.collection_id_by_detection_id[detection_id] = self.selected_collection_id_left
        self.detection_ids_by_collection_id[self.selected_collection_id_left].add(detection_id)
        self.detection_ids_by_collection_id[former_collection_id].remove(detection_id)
        if not self.detection_ids_by_collection_id[former_collection_id]:
            del self.detection_ids_by_collection_id[former_collection_id]

        self.show_detections()

    def transfer_detection_command(self, detection_id_str: str) -> str:
        try:
            self.transfer_detection(int(detection_id_str))
        except ValueError:
            return "Invalid detection id {!r}".format(detection_id_str)
        return "Success"

    def single_out_detection(self, detection_id: typing.Optional[int] = None) -> None:
        if self.selected_collection_id_left is None and detection_id is None:
            return

        if detection_id is None:
            for detection in self.shown_detections:
                if self.collection_id_by_detection_id[detection.id] == self.selected_collection_id_left:
                    detection_id = detection.id
                    former_collection_id = self.selected_collection_id_left
                    break
            else:
                return
        else:
            try:
                former_collection_id = self.collection_id_by_detection_id[detection_id]
            except KeyError:
                return

        collection_id = self.get_new_collection_id()
        self.detection_ids_by_collection_id[collection_id] = {detection_id}
        self.detection_ids_by_collection_id[former_collection_id].remove(detection_id)
        self.collection_id_by_detection_id[detection_id] = collection_id
        self.show_detections()

    def single_out_detection_command(self, detection_id_str: str) -> None:
        try:
            detection_id = int(detection_id_str)
        except ValueError:
            return
        self.single_out_detection(detection_id)

    def swap_showing_ids(self) -> None:
        self.showing_ids = not self.showing_ids
        self.show_detections()

    def show_last_search(self) -> None:
        if self.last_search is None:
            return
        left_detection_id, right_detection_id = self.last_search
        if left_detection_id in (x.id for frame in self.current_frames.frames for x in frame.detections):
            self.find(right_detection_id)
        else:
            self.find(left_detection_id)

    def load_pairs(self, filename) -> None:
        with open(filename, 'r') as f:
            data = f.read().strip().split('\n')
        self.loaded_pairs = (tuple(map(int, row.split())) for row in data if not row.startswith('# '))
        self.loaded_commented_rows = [row for row in data if row.startswith('# ')]

    def postpone_loaded_pair(self) -> None:
        if self.last_loaded_pair is not None:
            self.loaded_commented_rows.append("# {} {}".format(*self.last_loaded_pair))

    def show_next_loaded_pair(self) -> str:
        if self.loaded_pairs is None:
            return "No loaded pairs"
        try:
            self.last_loaded_pair = next(self.loaded_pairs)
        except StopIteration:
            self.last_loaded_pair = None
            return "No more pairs to process"
        left, right = self.last_loaded_pair
        self.selected_collection_id_right = self.collection_id_by_detection_id.get(right)
        result = self.find_and_select(left, SelectionType.Left)
        self.last_search = self.last_loaded_pair
        self.show_detections()
        return "Pair found" if result else "Pair not found"

    def save_loaded_pairs(self, filename, force=False) -> str:
        if not force and os.path.exists(filename):
            return "File {} already exists".format(shlex.quote(filename))
        with open(filename, 'w') as f:
            for comment in self.loaded_commented_rows:
                f.write(comment)
                f.write('\n')
            for pair in self.loaded_pairs:
                f.write("{} {}\n".format(*pair))

    def select_collection(self, collection_id, selection: SelectionType) -> None:
        if selection is SelectionType.Left:
            self.selected_collection_id_left = collection_id
        elif selection is SelectionType.Right:
            self.selected_collection_id_right = collection_id
        self.find_selected(selection, False)

    def delete_collection(self) -> None:
        collection_id = self.selected_collection_id_left
        if collection_id is None:
            return
        detection_ids = self.detection_ids_by_collection_id[collection_id]
        if self.selected_collection_id_right == self.selected_collection_id_left:
            self.selected_collection_id_right = None
        self.selected_collection_id_left = None
        del self.detection_ids_by_collection_id[collection_id]
        for detection_id in detection_ids:
            del self.collection_id_by_detection_id[detection_id]
        self.show_detections()

    def paste_collection(self) -> None:
        if self.detections_by_id is None:
            return

        if self.selected_collection_id_left is None:
            return

        self.background_image = PIL.Image.new('RGB', self.enclosing_dimensions)

        for detection_id in self.detection_ids_by_collection_id[self.selected_collection_id_left]:
            if detection_id not in self.detections_by_id:
                continue
            detection = self.detections_by_id[detection_id]
            crop = PIL.Image.fromarray(detection.image)
            camera_idx = [camera.id for camera in self.cameras].index(detection.frame.camera_id)
            frame_offset = self.dimension_offsets[camera_idx]
            if self.scale != 1:
                crop = crop.resize(tuple(reversed(np.int32(detection.dimensions * self.scale))), PIL.Image.ANTIALIAS)
            self.background_image.paste(crop, tuple(np.array((detection.position + frame_offset) * self.scale, np.int32)))
        self.update_image(self.background_image)
        self.showing_single_collection = True

    def setup_key_handler_signpost(self) -> None:
        self.basic_commands = {
            'Left': functools.partial(self.seek, -1),
            'Right': functools.partial(self.seek, 1),
            'Down': functools.partial(self.seek_until_unselected, -1),
            'Up': functools.partial(self.seek_until_unselected, 1),
            'Prior': functools.partial(self.seek, 10),
            'Next': functools.partial(self.seek, -10),
            'Home': functools.partial(self.set_frame, 0),
            'End': functools.partial(self.set_frame, -1),
        }
        self.commands_without_parameters = {
            'unify': self.unify,
            'hide': self.hide_collection,
            'all': self.swap_showing_unassigned,
            'quit': self.quit,
            'left': functools.partial(self.find_selected, SelectionType.Left),
            'right': functools.partial(self.find_selected, SelectionType.Right),
            'transfer': self.transfer_detection,
            'single_auto': self.single_out_detection,
            'id': self.swap_showing_ids,
            'memory': self.show_last_search,
            'postpone': self.postpone_loaded_pair,
            'next': self.show_next_loaded_pair,
            'clear': self.clear_view,
            'delete': self.delete_collection,
            'write': self.paste_collection,
        }
        self.short_commands_without_parameters = {key[0]: value
                                                  for key, value in self.commands_without_parameters.items()}
        self.commands_with_parameters = {
            'save': self.save,
            'save!': functools.partial(self.save, force=True),
            'upload': functools.partial(self.upload, loaded_only=False),
            'upload_loaded': functools.partial(self.upload, loaded_only=True),
            'find': self.find_command,
            'single': self.single_out_detection_command,
            'load_pairs': self.load_pairs,
            'save_pairs': self.save_loaded_pairs,
            'save_pairs!': functools.partial(self.save_loaded_pairs, force=True),
            'select_left': functools.partial(self.select_collection, selection=SelectionType.Left),
            'select_right': functools.partial(self.select_collection, selection=SelectionType.Right),
            'skip': lambda n_frames: self.seek(int(n_frames)),
        }
        assert len(self.short_commands_without_parameters) == len(self.commands_without_parameters)

    def key_handler(self, e):
        self.keys_since_quit += 1
        old_active_command = self.active_command

        if e.keysym in self.basic_commands:
            self.basic_commands[e.keysym]()
            return

        if self.active_command is not None:
            if e.keysym in ('Return', 'KP_Enter'):
                if ' ' in self.active_command:
                    command, parameters = self.active_command.split(' ', 1)
                else:
                    command = self.active_command
                    parameters = None
                if command in self.commands_without_parameters:
                    self.last_command_without_parameters = self.commands_without_parameters[command]
                    self.commands_without_parameters[command]()
                elif command in self.commands_with_parameters:
                    info = self.commands_with_parameters[command](parameters)
                    if info is not None:
                        self.logger.info(info)
                else:
                    self.logger.error("Unknown command {!r}".format(command))
                self.active_command = None
            elif e.keysym == 'BackSpace':
                if self.active_command:
                    self.active_command = self.active_command[:-1]
                else:
                    self.active_command = None
            elif e.keysym == 'Escape':
                self.active_command = None
            elif e.char in string.printable:
                self.active_command += e.char
            else:
                self.logger.error("Invalid key {!r}".format(e.keysym))
        else:
            if e.keysym in self.short_commands_without_parameters:
                self.last_command_without_parameters = self.short_commands_without_parameters[e.keysym]
                self.short_commands_without_parameters[e.keysym]()
            elif e.keysym == 'Escape':
                self.quit()
            elif e.keysym in ('colon', 'semicolon'):
                self.active_command = ''
            elif e.keysym == 'slash':
                self.active_command = 'find '
            else:
                self.logger.error("Unknown command {!r}".format(e.keysym))

        if old_active_command != self.active_command:
            self.show_detections()

    def button_handler(self, e):
        if self.showing_single_collection:
            self.showing_single_collection = False
            self.clear_view()
            return

        if e.num == 4:
            self.seek(1)
            return
        elif e.num == 5:
            self.seek(-1)
            return

        pos = np.array([e.x, e.y]) / self.scale
        for i_camera, dimensions in enumerate(self.dimensions):
            if pos[self.stack_dimension] - dimensions[self.stack_dimension] < 0:
                break
            pos[self.stack_dimension] -= dimensions[self.stack_dimension]
        camera_id = self.cameras[i_camera].id

        for detection in self.shown_detections:
            if (
                    detection.frame.camera_id == camera_id
                    and detection.left <= pos[0] < detection.right
                    and detection.top <= pos[1] < detection.bottom
            ):
                selected_collection_id = self.collection_id_by_detection_id[detection.id]
                break
        else:
            return
        if e.num == 1:
            self.selected_collection_id_left = selected_collection_id
        elif e.num == 3:
            self.selected_collection_id_right = selected_collection_id
        elif e.num == 2:
            if selected_collection_id in self.highlighted_collection_ids:
                self.highlighted_collection_ids.remove(selected_collection_id)
            else:
                self.highlighted_collection_ids.add(selected_collection_id)
        self.show_detections()

    def double_button_handler(self, e):
        if e.num not in (1,2,3):
            return
        if self.last_command_without_parameters is not None:
            self.last_command_without_parameters()


def main():
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)
    argument_parser.add_argument('-c', '--camera_ids', required=True, nargs="+", type=int,
                                 help="ID of camera(s) to explore")
    argument_parser.add_argument('-f', '--file',
                                 help="Name of the file to load trajectory model from")
    argument_parser.add_argument('-t',  '--trajectory', type=int, dest='trajectory_model_id',
                                 help="ID of base trajectory model")
    argument_parser.add_argument('-i', '--identity', type=int, dest='identity_model_id',
                                 help="ID of base identity model")
    argument_parser.add_argument('-p', '--preload', action='store_true',
                                 help="Preload all detection in initialization")
    argument_parser.add_argument('--fullscreen', '--fs', action='store_true',
                                 help='Run in fullscreen mode')
    argument_parser.add_argument('-s', '--scale', type=float, default=1,
                                 help="Scale the screen of the camera(s) by given factor")
    argument_parser.add_argument('-k', '--class', dest='class_', metavar="CLASS",
                                 help="If specified, show only detections of given class")
    argument_parser.add_argument('--start', dest='start_time_str', metavar="START_TIME",
                                 help="Query only frames from this timestamp onwards (relative to the first frame "
                                 "of this camera)")
    argument_parser.add_argument('--end', dest='end_time_str', metavar="END_TIME",
                                 help="Query only frames up to this timestamp (relative to the first frame of this "
                                      "camera)")
    parsed_arguments = argument_parser.parse_args()

    if (
        (parsed_arguments.trajectory_model_id is not None)
        + (parsed_arguments.identity_model_id is not None)
        + (parsed_arguments.file is not None)
    ) != 1:
        argument_parser.error("You need to specify exactly one of trajectory model id, identity model id or file")

    if len(parsed_arguments.camera_ids) > 1 and parsed_arguments.trajectory_model_id is not None:
        argument_parser.error("You can not select multiple cameras if you select trajectory model id")

    root = tkinter.Tk()
    app = AppFix(root, **vars(parsed_arguments))
    if parsed_arguments.fullscreen:
        root.attributes('-fullscreen', True)
    root.mainloop()


if __name__ == '__main__':
    main()
