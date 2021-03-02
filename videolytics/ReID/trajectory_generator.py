"""Collection of clustering algorithms (i.e. algorithms that create trajectories and identities)"""


import abc
import argparse
import collections
import datetime
import functools
import itertools
import logging
import numpy as np
import operator
import scipy.spatial.distance
import sqlalchemy
import sqlalchemy.orm
import tensorflow
import tensorflow_addons
import tqdm
import typing

from . import annotator
from . import custom_argparse
from . import database_connection as connection
from . import evaluate
from . import smooth_trajectory
from . import string_utils
from . import unionfind
from . import utils


class CollectionGenerator:
    """Base class for all identity/trajectory generators"""

    public_collection_generators: typing.Dict[str, typing.Type['CollectionGenerator']] = dict()
    argument_parser: typing.Optional[argparse.ArgumentParser] = None
    is_final: bool = False
    model: typing.Union[connection.TrajectoryModel, connection.IdentityModel] = None

    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None,
                 reidentification: bool = False, **kwargs):
        if options is not None and kwargs:
            raise RuntimeError("Creating collection generator {!r} with both: options and explicit parameters".format(
                self.__class__.__name__))

        self.reidentification = reidentification
        self.logger = logger
        if self.argument_parser is None:
            self.setup_argument_parser()

    def __init_subclass__(cls: typing.Type['CollectionGenerator'], **kwargs):
        if cls.is_final:
            kebab_name = string_utils.CaseTransformer.from_pascal(cls.__name__).to_kebab()
            CollectionGenerator.public_collection_generators[kebab_name] = cls

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        if argument_parser is None:
            argument_parser = argparse.ArgumentParser()
        cls.argument_parser = argument_parser
        return argument_parser

    @abc.abstractmethod
    def generate_collections(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None,
                             page_size: int = 5000, threads: int = 1,
                             last_frame: typing.Optional[datetime.timedelta] = None,
                             skip_annotation: bool = False) -> None:
        pass

    @staticmethod
    def initialize_camera_setup(camera_ids: typing.Iterable[int]) -> connection.CameraSetup:
        if not isinstance(camera_ids, list):
            camera_ids = list(camera_ids)
        camera_ids_set = set(camera_ids)
        camera_setups = (
            connection.session
            .query(connection.CameraSetup)
            .options(
                sqlalchemy.orm
                .joinedload(connection.CameraSetup.camera_setup_assignments)
            )
        )
        for camera_setup in camera_setups:
            if {assignment.camera_id for assignment in camera_setup.camera_setup_assignments} == camera_ids_set:
                return camera_setup
        else:
            camera_setup = connection.CameraSetup(
                description="Setup for cameras {}".format(' '.join(map(str, sorted(camera_ids_set))))
            )
            connection.session.add(camera_setup)
            connection.session.flush()
            for camera_id in camera_ids:
                connection.session.add(connection.CameraSetupAssignment(
                    camera_setup_id=camera_setup.id,
                    camera_id=camera_id,
                ))
            connection.session.flush()
            return camera_setup


class CollectionGeneratorWithoutAnnotator(CollectionGenerator):
    """Base class for annotators that do not require annotators"""
    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None,
                 reidentification: bool = False, **kwargs):
        super().__init__(logger, options, reidentification)
        if options is not None:
            kwargs = vars(self.argument_parser.parse_args(options))

        self.parameters = kwargs
        self.option_string = custom_argparse.argument_string(self.argument_parser, kwargs)

        model_parameters = {
            'generator_name': string_utils.CaseTransformer.from_pascal(self.__class__.__name__).to_kebab(),
            'options': self.option_string,
        }

        type_str, ModelClass = ('Identity', connection.IdentityModel) if reidentification else ('Trajectory', connection.TrajectoryModel)

        model = (
            connection.session
            .query(ModelClass)
            .filter_by(**model_parameters)
            .one_or_none()
        )
        if model is None:
            model = ModelClass(**model_parameters)
            connection.session.add(model)
            connection.session.flush()
            logger.info("{} model {!r} does not exists, creating new one".format(type_str, self))
        else:
            logger.info("{} model {!r} exists, using that one".format(type_str, self))
        self.model = model

    def __repr__(self):
        return '{}[{}]'.format(self.__class__.__name__, self.option_string)


class CollectionGeneratorWithAnnotator(CollectionGenerator):
    """Base class for generators that use annotation with feature vectors"""

    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None,
                 reidentification: bool = False, **kwargs):
        super().__init__(logger, options, reidentification, **kwargs)

        if options is not None:
            kwargs = vars(self.argument_parser.parse_args(options))

        annotator_string = kwargs.pop('annotator')
        annotator_parameters = kwargs.pop('annotator_arguments')
        AnnotatorClass: typing.Type[annotator.Annotator] = annotator.Annotator.public_annotators[annotator_string]
        annotator_logger = utils.get_logger('Annotator', self.logger.level)
        self.annotator = AnnotatorClass(logger=annotator_logger, options=annotator_parameters)

        self.parameters = kwargs
        self.option_string = custom_argparse.argument_string(self.argument_parser, kwargs)

        model_parameters = {
            'generator_name': string_utils.CaseTransformer.from_pascal(self.__class__.__name__).to_kebab(),
            'options': self.option_string,
            'feature_type_id': self.annotator.feature_type.id,
        }

        type_str, ModelClass = ('Identity', connection.IdentityModel) if reidentification else ('Trajectory', connection.TrajectoryModel)

        model = (
            connection.session
            .query(ModelClass)
            .filter_by(**model_parameters)
            .one_or_none()
        )
        if model is None:
            model = ModelClass(**model_parameters)
            connection.session.add(model)
            connection.session.flush()
            logger.info("{} model {!r} does not exists, creating new one".format(type_str, self))
        else:
            logger.info("{} model {!r} exists, using that one".format(type_str, self))
        self.model = model

    def __repr__(self):
        return "{}[{!r} ; {}]".format(self.__class__.__name__, self.annotator, self.option_string)

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        argument_parser = super().setup_argument_parser(argument_parser)
        argument_parser.add_argument('annotator', choices=annotator.Annotator.public_annotators.keys(),
                                     help="Annotator to use for the annotation")
        argument_parser.add_argument('annotator_arguments', nargs=argparse.REMAINDER,
                                     help="Additional arguments for chosen annotator")
        return argument_parser

    def apply_detection_filters(self, query: sqlalchemy.orm.Query, camera_ids: typing.List[int],
                                class_: typing.Optional[str] = None,
                                last_frame: typing.Optional[datetime.timedelta] = None) -> sqlalchemy.orm.Query:
        query = (
            query
            .filter(connection.FeatureDescriptor.feature_type_id == self.annotator.feature_type.id)
            .filter(connection.Frame.camera_id.in_(camera_ids))
        )

        if class_ is not None:
            query = query.filter(connection.Detection.class_ == class_.upper())

        if last_frame is not None:
            first_timestamp = (
                connection.session
                .query(sqlalchemy.func.min(connection.Frame.timestamp))
                .filter(connection.Frame.camera_id.in_(camera_ids))
            ).scalar()
            query = query.filter(connection.Frame.timestamp < first_timestamp + last_frame)
        return query

    def query_detections(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None, threads: int = 1,
                         last_frame: typing.Optional[datetime.timedelta] = None,
                         skip_annotation: bool = False) -> sqlalchemy.orm.Query:
        if not skip_annotation:
            self.annotator.annotate(camera_ids, class_, threads=threads)

        query = (
            connection.session
            .query(connection.FeatureDescriptor)
            .join(connection.Detection)
            .join(connection.Frame)
            .options(
                sqlalchemy.orm
                .joinedload(connection.FeatureDescriptor.detection)
                .joinedload(connection.Detection.frame),
                # sqlalchemy.orm.contains_eager(connection.FeatureDescriptor.detection),
                # sqlalchemy.orm.contains_eager(connection.Detection.frame),
                sqlalchemy.orm.defaultload(connection.FeatureDescriptor.detection).defer(connection.Detection.crop)
            )
            .order_by(connection.Frame.timestamp, connection.Frame.id, connection.Detection.id)
        )
        return self.apply_detection_filters(query, camera_ids, class_, last_frame)

    def count_detections(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None, threads: int = 1,
                         last_frame: typing.Optional[datetime.timedelta] = None) -> int:
        query = (
            connection.session
            .query(sqlalchemy.func.count(connection.FeatureDescriptor.id))
            .join(connection.Detection)
            .join(connection.Frame)
        )
        query = self.apply_detection_filters(query, camera_ids, class_, last_frame)
        return query.scalar()

    @abc.abstractmethod
    def generate_collections(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None,
                             page_size: int = 5000, threads: int = 1,
                             last_frame: typing.Optional[datetime.timedelta] = None,
                             skip_annotation: bool = False) -> None:
        pass


class FeatureWithUnpacked(typing.NamedTuple):
    feature_descriptor: connection.FeatureDescriptor
    unpacked_value: np.ndarray


class Check:
    """Class that allows for combining functions that returns bool and then evaluate the function on the output"""
    checker: typing.Callable[[FeatureWithUnpacked, FeatureWithUnpacked], bool]

    def __init__(self, checker: typing.Callable[[FeatureWithUnpacked, FeatureWithUnpacked], bool]):
        self.checker = checker

    def __and__(self, other):
        return self.combine(operator.and_, other)

    def __or__(self, other):
        return self.combine(operator.or_, other)

    def combine(self, op, other):
        if isinstance(other, Check):
            return Check(lambda *args: op(self.checker(*args), other.checker(*args)))
        if callable(other):
            return Check(lambda *args: op(self.checker(*args), other(*args)))
        raise RuntimeError("Cannot combine Check with {!r}".format(other))

    def __call__(self, *args):
        return self.checker(*args)


class CollectionGeneratorWithAnnotatorAndTrajectories(CollectionGeneratorWithAnnotator):
    """Base class for annotators that use both -- feature vectors and a priori trajectories"""

    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None,
                 reidentification: bool = False, **kwargs):
        if not reidentification:
            raise RuntimeError("Collection Generator {!r} has to be run with reidentification flag".format(
                self.__class__.__name__
            ))
        super().__init__(logger, options, reidentification, **kwargs)

        if options is not None:
            kwargs = vars(self.argument_parser.parse_args(options))

        trajectory_model_id = kwargs.pop('trajectory_model_id')
        try:
            self.trajectory_model = (
                connection.session
                .query(connection.TrajectoryModel)
                .filter_by(id=trajectory_model_id)
                .one()
            )
        except sqlalchemy.orm.exc.NoResultFound:
            raise RuntimeError("No trajectory model with id {}".format(trajectory_model_id))

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        argument_parser = super().setup_argument_parser(argument_parser)
        argument_parser.add_argument('-t', '--trajectory-model-id', type=int, required=True,
                                     help="ID of trajectory model to use for reidentification")
        return argument_parser

    def query_detection_data(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None) -> \
            typing.Iterator[typing.Tuple[connection.Detection, connection.FeatureDescriptor, connection.Trajectory]]:
        trajectory_model_id = self.trajectory_model.id
        feature_type_id = self.annotator.feature_type.id

        trajectory_query = (
            connection.session
            .query(connection.Trajectory, connection.TrajectoryDetection)
            .join(connection.TrajectoryDetection)
            .filter(
                (connection.Trajectory.trajectory_model_id == trajectory_model_id)
                & (connection.Trajectory.camera_id.in_(camera_ids))
            )
            .with_labels()
            .subquery()
        )

        aliased_trajectory = sqlalchemy.orm.aliased(connection.Trajectory, trajectory_query)
        aliased_trajectory_detection = sqlalchemy.orm.aliased(connection.TrajectoryDetection, trajectory_query)

        query = (
            connection.session
            .query(connection.Detection, connection.FeatureDescriptor, aliased_trajectory)
            .join(connection.Frame, connection.Detection.frame_id == connection.Frame.id)
            .join(connection.FeatureDescriptor,
                  (connection.FeatureDescriptor.detection_id == connection.Detection.id)
                  & (connection.FeatureDescriptor.feature_type_id == feature_type_id))
            .join(aliased_trajectory_detection)
            .options(sqlalchemy.orm.defer(connection.Detection.crop))
            .filter(connection.Frame.camera_id.in_(camera_ids))
            .order_by(connection.Frame.timestamp, connection.Detection.id)
        )

        if class_ is not None:
            query = query.filter(connection.Detection.class_ == class_.upper())

        return query


class TrajectoryMerge(CollectionGeneratorWithAnnotatorAndTrajectories):
    '''Generator based on a priori trajectories that are marged based on feature vectors'''
    is_final: bool = True

    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None,
                 reidentification: bool = False, **kwargs):
        super(TrajectoryMerge, self).__init__(logger, options, reidentification, **kwargs)

        distance_functions = {
            'euclidean': scipy.spatial.distance.euclidean,
            'cosine': scipy.spatial.distance.cosine,
            'manhattan': scipy.spatial.distance.cityblock,
        }

        self.used_distance_functions: typing.List[typing.List[typing.Callable[[np.array, np.array], bool]]] = [
            [(lambda x, y: func(x, y) < th) for th in self.parameters[name]]
            for name, func in distance_functions.items()
            if self.parameters[name] is not None
        ]


    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        argument_parser = super().setup_argument_parser(argument_parser)
        argument_parser.add_argument('--cosine', '-c', type=float, nargs=2, default=None,
                                     help="Threshold for cosine distance, first threshold is used for representants"
                                          " the second for merging trajectories")
        argument_parser.add_argument('--euclidean', '-e', type=float, nargs=2, default=None,
                                     help = "Threshold for euclidean distance, first threshold is used for representants"
                                            " the second for merging trajectories")
        argument_parser.add_argument('--manhattan', '-m', type=float, nargs=2, default=None,
                                     help = "Threshold for manhattan distance, first threshold is used for representants"
                                            " the second for merging trajectories")
        return argument_parser

    def generate_collections(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None,
                             page_size: int = 5000, threads: int = 1,
                             last_frame: typing.Optional[datetime.timedelta] = None,
                             skip_annotation: bool = False) -> None:
        query = self.query_detection_data(camera_ids, class_)
        n_detections = query.count()
        paged_query = utils.smooth_paged_query(query, page_size)

        detection_id_and_features_by_trajectory_id: typing.Dict[int, typing.List[typing.Tuple[int, np.array]]] = dict()
        self.logger.debug("Pre-processing detections")
        for detection, feature_descriptor, trajectory in tqdm.tqdm(paged_query, desc="Pre-processing detections",
                                                                   total=n_detections):
            data_list = detection_id_and_features_by_trajectory_id.setdefault(trajectory.id, list())
            data_list.append((detection.id, np.frombuffer(feature_descriptor.value, dtype=np.float32)))

        self.logger.debug("Generating representative subsets")
        representative_features = dict()
        for trajectory_id, data in tqdm.tqdm(detection_id_and_features_by_trajectory_id.items(),
                                             desc="Generating representative subsets",
                                             total=len(detection_id_and_features_by_trajectory_id)):
            representants_per_dist_function = list()
            for i, dist_func in enumerate(self.used_distance_functions):
                representants = list()
                for _, feature in data:
                    if not any(dist_func[0](rep, feature) for rep in representants):
                        representants.append(feature)
                representants_per_dist_function.append(representants)
            representative_features[trajectory_id] = representants_per_dist_function

        avg_representants = utils.avg([
            len(per_dist_data)
            for trajectory_data in representative_features.values()
            for per_dist_data in trajectory_data
        ])
        self.logger.debug("Total number of trajectories: {}".format(len(representative_features)))
        self.logger.debug("Average number of representants: {}".format(avg_representants))

        merged_trajectories: unionfind.UnionFind[int] = unionfind.UnionFind(representative_features.keys())
        self.logger.debug("Unifying trajectories")
        for trajectory_id_a, trajectory_id_b in tqdm.tqdm(
                itertools.combinations(representative_features.keys(), 2),
                total=(len(representative_features) * (len(representative_features) - 1)) // 2,
                desc="Possible trajectory combinations",
        ):
            if all(
                any(
                    dist_func[1](representant_a, representant_b)
                    for representant_a, representant_b in zip(representants_a, representants_b)
                ) for representants_a, representants_b, dist_func in zip(
                    representative_features[trajectory_id_a],
                    representative_features[trajectory_id_b],
                    self.used_distance_functions,
                )
            ):
                merged_trajectories.union(trajectory_id_a, trajectory_id_b)

        camera_setup_id = self.initialize_camera_setup(camera_ids).id
        self.logger.debug("Camera setup id identified as {}".format(camera_setup_id))

        def create_new_identity():
            identity = connection.Identity(
                identity_model_id=self.model.id,
                camera_setup_id=camera_setup_id,
            )
            connection.session.add(identity)
            connection.session.flush()
            return identity

        identities_by_root_trajectory_id = collections.defaultdict(create_new_identity)
        identity_detections_for_commit = list()
        self.logger.debug("Saving detections")
        progress_bar = tqdm.tqdm(desc="Saving detections", total=n_detections)
        for trajectory_id, trajectory_data in detection_id_and_features_by_trajectory_id.items():
            for detection_id, feature in trajectory_data:
                identity = identities_by_root_trajectory_id[merged_trajectories.find(trajectory_id)]
                identity_detections_for_commit.append(connection.IdentityDetection(
                    identity_id=identity.id,
                    detection_id=detection_id,
                ))
                progress_bar.update(1)
        self.logger.debug("Committing detections")
        connection.session.bulk_save_objects(identity_detections_for_commit)
        connection.session.commit()
        self.logger.info("New detections committed")


class Direct(CollectionGeneratorWithAnnotator):
    """Trajectory generator that use direct approach for generation of identities/trajectories"""
    is_final: bool = True

    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None,
                 reidentification: bool = False, **kwargs):
        super().__init__(logger, options, reidentification, **kwargs)
        self.time_window = datetime.timedelta(seconds=self.parameters['time_window'])

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        argument_parser = super().setup_argument_parser(argument_parser)
        argument_parser.add_argument('-t', '--time-window', type=float, required=True,
                                     help='Length of time window in seconds to allow trajectories to merge')
        argument_parser.add_argument('--spatial-time-window', type=float, default=None,
                                     help="Shorter time window that is used for comparison of metadata")
        argument_parser.add_argument('-l', '--relative-displacement', type=float, default=None)
        argument_parser.add_argument('-d', '--displacement', type=float, default=None)
        argument_parser.add_argument('--iou', type=float)
        argument_parser.add_argument('-c', '--cosine', type=float)
        argument_parser.add_argument('-e', '--euclidean', type=float)
        argument_parser.add_argument('-m', '--manhattan', type=float)
        return argument_parser

    def generate_collections(self, camera_ids: typing.List[int], class_: typing.Optional[str] = None,
                             page_size: int = 5000, threads: int = 1,
                             last_frame: typing.Optional[datetime.timedelta] = None,
                             skip_annotation: bool = False) -> None:
        query = self.query_detections(camera_ids, class_, threads, last_frame, skip_annotation)
        memory: typing.Deque[FeatureWithUnpacked] = collections.deque()

        self.logger.debug("Checking for detections fitting the parameters")
        if query.first() is None:
            self.logger.error("There seems to be no detections fitting the query")
            return

        self.logger.debug("Establishing checks fitting given parameters")

        check = Check(lambda *args: True)

        if self.parameters['relative_displacement'] is not None:
            def relative_displacement_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                historical_detection: connection.Detection = historical.feature_descriptor.detection
                present_detection: connection.Detection = present.feature_descriptor.detection
                if historical_detection.frame.camera_id != present_detection.frame.camera_id:
                    return False

                displacement = np.linalg.norm(historical_detection.center - present_detection.center)
                dimension_sum = sum([
                    historical_detection.height,
                    historical_detection.width,
                    present_detection.height,
                    present_detection.width,
                ])
                relative_displacement = displacement / dimension_sum * 4
                return relative_displacement <= self.parameters['relative_displacement']
            check = check & relative_displacement_check

        if self.parameters['displacement'] is not None:
            def displacement_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                historical_detection: connection.Detection = historical.feature_descriptor.detection
                present_detection: connection.Detection = present.feature_descriptor.detection
                if historical_detection.frame.camera_id != present_detection.frame.camera_id:
                    return False

                displacement = np.linalg.norm(historical_detection.center - present_detection.center)
                return displacement <= self.parameters['displacement']
            check = check & displacement_check

        if self.parameters['iou'] is not None:
            def iou_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                historical_detection: connection.Detection = historical.feature_descriptor.detection
                present_detection: connection.Detection = present.feature_descriptor.detection
                if historical_detection.frame.camera_id != present_detection.frame.camera_id:
                    return False

                left = max(historical_detection.left, present_detection.left)
                right = min(historical_detection.right, present_detection.right)
                top = max(historical_detection.top, present_detection.top)
                bottom = min(historical_detection.bottom, present_detection.bottom)

                width = right - left
                height = bottom - top

                if width <= 0 or height <= 0:
                    return False

                intersection = width * height
                historical = historical_detection.width * historical_detection.height
                present = present_detection.width * present_detection.height
                combined = historical + present
                union = combined - historical
                return intersection >= self.parameters['iou'] * union
            check = check & iou_check

        if self.parameters['spatial_time_window'] is not None:
            time_threshold = datetime.timedelta(seconds=self.parameters['spatial_time_window'])
            def spatial_time_window_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                return present.feature_descriptor.detection.frame.timestamp - historical.feature_descriptor.detection.frame.timestamp < time_threshold
            check = check & spatial_time_window_check

        if self.parameters['cosine'] is not None:
            def cosine_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                return scipy.spatial.distance.cosine(historical.unpacked_value, present.unpacked_value) <= self.parameters['cosine']
            check = check | cosine_check

        if self.parameters['manhattan'] is not None:
            def manhattan_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                return scipy.spatial.distance.cityblock(historical.unpacked_value, present.unpacked_value) <= self.parameters['cosine']
            check = check | manhattan_check

        if self.parameters['euclidean'] is not None:
            def euclidean_check(historical: FeatureWithUnpacked, present: FeatureWithUnpacked) -> bool:
                return scipy.spatial.distance.euclidean(historical.unpacked_value, present.unpacked_value) <= self.parameters['cosine']
            check = check | euclidean_check

        camera_ids_iterator = iter(camera_ids)
        if self.reidentification:
            Collection = connection.Identity
            CollectionDetection = connection.IdentityDetection

            camera_setup = self.initialize_camera_setup(camera_ids)
            setup_id = camera_setup.id

            create_collection = functools.partial(
                connection.Identity,
                identity_model_id=self.model.id,
                camera_setup_id=setup_id,
            )

            def create_collection_detection(collection_id, detection_id):
                return connection.IdentityDetection(
                    identity_id=collection_id,
                    detection_id=detection_id,
                )
        else:
            Collection = connection.Trajectory
            CollectionDetection = connection.TrajectoryDetection
            camera_id = next(camera_ids_iterator)
            try:
                next(camera_ids_iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Supplying multiple camera ids without reidentification parameter")

            create_collection = functools.partial(
                connection.Trajectory,
                trajectory_model_id=self.model.id,
                camera_id=camera_id,
            )

            def create_collection_detection(collection_id, detection_id):
                return connection.TrajectoryDetection(
                    trajectory_id=collection_id,
                    detection_id=detection_id,
                )

        self.logger.debug("Checks established, calculating number of detections")
        total = self.count_detections(camera_ids, class_, threads, last_frame)
        if page_size > 0:
            query = utils.smooth_paged_query(query, page_size)
        self.logger.debug("Detections counted, started processing them")

        final_collections: unionfind.UnionFind[id] = unionfind.UnionFind()
        iterator = tqdm.tqdm(query, total=total, desc="Processing detections")
        for feature_descriptor in iterator:
            feature_descriptor: connection.FeatureDescriptor
            unpacked_value = np.frombuffer(feature_descriptor.value, dtype=np.float32)
            this = FeatureWithUnpacked(feature_descriptor, unpacked_value)
            while memory and feature_descriptor.detection.frame.timestamp - memory[0].feature_descriptor.detection.frame.timestamp > self.time_window:
                memory.popleft()
            final_collections.add(feature_descriptor.detection_id)
            for other in memory:
                if check(other, this):
                    final_collections.union(other.feature_descriptor.detection_id, feature_descriptor.detection_id)

            memory.append(this)

        final_collections_flatten = [list(x) for x in final_collections.get_sets()]
        db_collections = [create_collection() for _ in final_collections_flatten]
        connection.session.add_all(db_collections)
        connection.session.flush()
        detections_to_commit: typing.List[CollectionDetection] = list()
        for detection_ids, collection in zip(final_collections_flatten, db_collections):
            detections_to_commit.extend(create_collection_detection(
                collection_id=collection.id,
                detection_id=detection_id,
            ) for detection_id in detection_ids)

        self.logger.debug("Collections created, committing into database")
        connection.session.bulk_save_objects(detections_to_commit)
        connection.session.commit()
        self.logger.info("Collections committed; total number of trajectories: {}".format(len(final_collections_flatten)))


def main(argument_string: typing.Optional[str] = None):
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)

    argument_parser.add_argument('-c', '--cameras', nargs="+", required=True,
                                 help='ID of camera to annotate')
    argument_parser.add_argument('--class', type=str, dest='class_', metavar='CLASS',
                                 help='Class to annotate (e.g. PERSON), if omitted all classes are annotated')
    argument_parser.add_argument('-p', '--page-size', type=int, default=5000,
                                 help='Number of detection requested from database at one time. Setting this to low '
                                      'number decreases local RAM usage but increases the number of database requests. '
                                      'To disable paging altogether, set this to 0')
    argument_parser.add_argument('-j', '--jobs', type=int, default=1,
                                 help='Number of jobs (threads) to use')
    argument_parser.add_argument('-g', '-e', '--golden-trajectory-model', '--evaluate', type=int, default=None,
                                 help="Evaluate created trajectory with given trajectory model as golden trajectory "
                                      "model")
    argument_parser.add_argument('-s', '--smooth', action="store_true",
                                 help="Smooth trajectories after creation")
    argument_parser.add_argument('--end',
                                 help="Timestamp of last frame to annotate (diff w.r.t. the first frame)")
    argument_parser.add_argument('collection_generator',
                                 choices=CollectionGenerator.public_collection_generators.keys(),
                                 help="Generator to use for generating trajectories")
    argument_parser.add_argument('-r', '--reidentification', action='store_true',
                                 help="Employ more advanced algorithm allowing joining trajectories across multiple "
                                      "cameras; if set, identities will be created instead of trajectories")
    argument_parser.add_argument('--skip-annotation', action='store_true',
                                 help="Skip the annotation with the provided annotator (i.e. presume the detections are"
                                      " already annotated")
    argument_parser.add_argument('collection_generator_arguments', nargs=argparse.REMAINDER,
                                 help="Additional parameters for selected trajectory generator")


    parsed_arguments = argument_parser.parse_args(argument_string)
    CollectionGeneratorClass = CollectionGenerator.public_collection_generators[parsed_arguments.collection_generator]
    logger = utils.get_logger('Collection Generator', parsed_arguments.log)

    if not parsed_arguments.reidentification and len(parsed_arguments.cameras) > 1:
        argument_parser.error("Multiple cameras are allowed only with reidentification option")

    collection_generator: CollectionGenerator = CollectionGeneratorClass(
        logger=logger,
        reidentification=parsed_arguments.reidentification,
        options=parsed_arguments.collection_generator_arguments,
    )
    collection_generator.generate_collections(
        camera_ids=[int(i) for i in parsed_arguments.cameras],
        class_=parsed_arguments.class_,
        page_size=parsed_arguments.page_size,
        threads=parsed_arguments.jobs,
        last_frame=utils.parse_timedelta(parsed_arguments.end),
        skip_annotation=parsed_arguments.skip_annotation,
    )

    logger.info("Created collection model {!r} has id {}".format(
        collection_generator, collection_generator.model.id))

    if parsed_arguments.smooth:
        for camera in parsed_arguments.cameras:
            smooth_trajectory.smooth(camera, collection_generator.model.id, parsed_arguments.page_size,
                                     parsed_arguments.log)

    if parsed_arguments.golden_trajectory_model:
        evaluate.evaluate(collection_generator.model.id, parsed_arguments.golden_trajectory_model,
                          parsed_arguments.page_size, parsed_arguments.log)


if __name__ == '__main__':
    with tensorflow.device('/cpu:0'):
        main()
