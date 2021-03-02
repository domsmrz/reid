"""All the classes for annotation (feature vector generation)"""


import itertools
from abc import ABC

import numpy as np
import abc
import logging
import cv2
import os
import random

import tensorflow
import tqdm
import matplotlib.colors
import argparse
import typing
import scipy.stats
import multiprocessing
import functools
import sqlalchemy.orm
import datetime
import hashlib

from . import utils
from . import string_utils
from . import database_connection as connection
from . import custom_argparse

import tensorflow.keras.applications.resnet50
import tensorflow.keras.applications.mobilenet_v2
import tensorflow.keras.models
import tensorflow.keras
import tensorflow_addons


class TripletLossTrainingSequence(tensorflow.keras.utils.Sequence):
    """
    Tensorflow sequence that allows for training with triplet loss on the Detections. Generates such batches
    that with multiple detection per identity to allow suitable selection of triplets
    """

    def __init__(self, ground_truth_trajectory_id: int, camera_id: int,
                 training_start_offset: typing.Optional[datetime.timedelta] = None,
                 training_end_offset: typing.Optional[datetime.timedelta] = None,
                 class_: typing.Optional[str] = None,
                 preprocessing_function: typing.Callable[[np.array], np.array] = lambda x: x,
                 minimum_trajectory_length: int = 4, seed: int = 42,
                 batch_size: int = 32, interwoven_trajectories: int = 4,
                 maximum_increase_interwoven_trajectories: int = 2,
                 shuffle_batches: bool = False):

        self.batch_size = batch_size
        self.interwoven_trajectories = interwoven_trajectories
        self.current_interwoven_trajectories = interwoven_trajectories - 1
        self.maximum_increase_interwoven_trajectories = maximum_increase_interwoven_trajectories
        self.random_generator = random.Random(seed)
        self.shuffle_batches = shuffle_batches

        frame_condition = connection.Frame.id == connection.Detection.frame_id
        if training_start_offset is not None or training_end_offset is not None:
            camera_start_time = (
                connection.session
                .query(sqlalchemy.func.min(connection.Frame.timestamp))
                .filter(connection.Frame.camera_id == camera_id)
                .scalar()
            )

            if training_start_offset is not None:
                training_start = camera_start_time + training_start_offset
                frame_condition &= connection.Frame.timestamp >= training_start
            if training_end_offset is not None:
                training_end = camera_start_time + training_end_offset
                frame_condition &= connection.Frame.timestamp < training_end

        golden_trajectory_query = (
            connection.session
            .query(connection.Detection, connection.Trajectory.id)
            .select_from(connection.Trajectory)
            .join(connection.TrajectoryDetection)
            .join(connection.Detection)
            .join(connection.Frame, frame_condition)
            .filter(connection.Trajectory.trajectory_model_id == ground_truth_trajectory_id)
            .filter(connection.Frame.camera_id == camera_id)
            .order_by(connection.Detection.id.desc())
        )

        if class_ is not None:
            golden_trajectory_query = golden_trajectory_query.filter(connection.Detection.class_ == class_)

        data_dict = dict()
        for detection, trajectory_id in golden_trajectory_query:
            processed_image = preprocessing_function(detection.image)
            trajectory_bucket = data_dict.setdefault(trajectory_id, [])
            trajectory_bucket.append(processed_image)

        data_dict = {k: v for k, v in data_dict.items() if len(v) >= minimum_trajectory_length}
        self.data = sorted(data_dict.items(), key=lambda item: (-len(item[1]), item[0]))
        self.batches = None
        self.prepare_epoch()

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate_batches(self) -> typing.Iterator[np.array]:
        for _, trajectory_data in self.data:
            self.random_generator.shuffle(trajectory_data)
        trajectory_iterator = ((trajectory_id, iter(trajectory_data)) for trajectory_id, trajectory_data in self.data)
        exhausted_trajectory_iterator = False
        self.current_interwoven_trajectories += 1
        if self.current_interwoven_trajectories > self.interwoven_trajectories + self.maximum_increase_interwoven_trajectories:
            self.current_interwoven_trajectories = self.interwoven_trajectories

        active_trajectories = list(itertools.islice(trajectory_iterator, self.current_interwoven_trajectories))
        active_trajectory_index = 0
        active_batch_features = list()
        active_batch_targets = list()

        while active_trajectories:
            if active_trajectory_index >= len(active_trajectories):
                active_trajectory_index = 0
            active_trajectory_id, active_trajectory_data = active_trajectories[active_trajectory_index]
            try:
                next_detection = next(active_trajectory_data)
            except StopIteration:
                if exhausted_trajectory_iterator:
                    active_trajectories.pop(active_trajectory_index)
                else:
                    try:
                        active_trajectories[active_trajectory_index] = next(trajectory_iterator)
                    except StopIteration:
                        exhausted_trajectory_iterator = True
                        active_trajectories.pop(active_trajectory_index)
            else:
                active_batch_features.append(next_detection)
                active_batch_targets.append(active_trajectory_id)

                if len(active_batch_targets) == self.batch_size:
                    yield np.stack(active_batch_features), np.stack(active_batch_targets)
                    active_batch_features = list()
                    active_batch_targets = list()
                active_trajectory_index += 1

        if active_batch_targets:
            yield np.stack(active_batch_features), np.stack(active_batch_targets)

    def prepare_epoch(self):
        self.batches = list(self.generate_batches())
        if self.shuffle_batches:
            self.random_generator.shuffle(self.batches)

    def on_epoch_end(self):
        self.prepare_epoch()


def descriptor_generator(feature_type_id: int, annotate: typing.Callable[[connection.Detection], bytes],
                         detection: connection.Detection, feature_descriptor: connection.FeatureDescriptor)\
                         -> typing.Optional[connection.FeatureDescriptor]:
    value = annotate(detection)
    if value is None:
        return None
    if feature_descriptor is None:
        feature_descriptor = connection.FeatureDescriptor(
            feature_type_id=feature_type_id,
            detection_id=detection.id
        )
    feature_descriptor.value = value
    return feature_descriptor


class Annotator(ABC):
    '''
    Base class of all the annotators
    '''

    public_annotators: typing.Dict[str, typing.Type['Annotator']] = dict()
    argument_parser: typing.Optional[argparse.ArgumentParser] = None
    is_public: bool = False

    def __init__(self, logger: logging.Logger, options: typing.Optional[typing.List[str]] = None, **kwargs):
        if options is not None and kwargs:
            raise RuntimeError("Creating annotator {!r} with both: options and explicit parameters".format(
                self.__class__.__name__))

        self.logger = logger
        if self.argument_parser is None:
            self.setup_argument_parser()

        if options is not None:
            kwargs = vars(self.argument_parser.parse_args(options))

        self.parameters = kwargs
        # option string stored in the database
        self.option_string = custom_argparse.argument_string(self.argument_parser, kwargs)
        feature_type_parameters = {
            'annotator_name': string_utils.CaseTransformer.from_pascal(self.__class__.__name__).to_kebab(),
            'options': self.option_string,
        }
        feature_type = connection.session.query(connection.FeatureType).filter_by(**feature_type_parameters).one_or_none()
        if feature_type is None:
            logger.info("Feature type {!r} does not exits, creating new one".format(self))
            feature_type = connection.FeatureType(**feature_type_parameters)
            connection.session.add(feature_type)
            connection.session.flush()
        else:
            logger.info("Feature type {!r} exists, using that one".format(self))
        self.feature_type = feature_type
        logger.debug("Established annotator {!r}".format(self))

    def __init_subclass__(cls: typing.Type['Annotator'], **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.is_public:
            kebab_class_name = string_utils.CaseTransformer.from_pascal(cls.__name__).to_kebab()
            Annotator.public_annotators[kebab_class_name] = cls

    def __repr__(self):
        return "{}[{}]".format(self.__class__.__name__, self.option_string)

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) ->\
            argparse.ArgumentParser:
        '''Method to define parameters for sepecific annotator'''
        if argument_parser is None:
            argument_parser = argparse.ArgumentParser()
        cls.argument_parser = argument_parser
        return argument_parser

    def annotate(self, camera_ids: typing.List[int], class_: typing.Optional[str], page_size: int = 5000, threads: int = 1,
                 force_update: bool = False, end: typing.Optional[datetime.timedelta] = None,
                 process_pool: typing.Optional[multiprocessing.Pool] = None,
                 by_identity_model: typing.Optional[int] = None) -> None:
        '''
        Main annotation method. Queries detection from the database and commits corresponding feature vectors.
        '''
        if threads > 1 and process_pool is None:
            raise RuntimeError("Annotator {!r} does not support multiple jobs".format(self))

        detection_query = (
            connection.session
            .query(connection.Detection, connection.FeatureDescriptor)
            .join(connection.Frame)
            .outerjoin(
                connection.FeatureDescriptor,
                (connection.Detection.id == connection.FeatureDescriptor.detection_id)
                & (connection.FeatureDescriptor.feature_type_id == self.feature_type.id)
            )
            .filter(connection.Frame.camera_id.in_(camera_ids))
            .order_by(connection.Detection.id)
        )

        if by_identity_model is not None:
            detection_query = detection_query.join(connection.IdentityDetection).join(connection.Identity).filter(connection.Identity.identity_model_id == by_identity_model)

        if end is not None:
            first_frame = (
                connection.session.query(sqlalchemy.func.min(connection.Frame.timestamp))
                .filter(connection.Frame.camera_id.in_(camera_ids))
                .scalar()
            )
            end_timestamp = first_frame + end
            detection_query = detection_query.filter(connection.Frame.timestamp <= end_timestamp)

        if class_ is not None:
            detection_query = (
                detection_query
                .filter(connection.Detection.class_ == class_.upper())
            )

        self.logger.debug("Querying first detection form the database")
        if detection_query.first() is None:
            self.logger.debug("There seems to be no detections fitting the parameters, skipping annotation")
            return

        if not force_update:
            detection_query = detection_query.filter(connection.FeatureDescriptor.id.is_(None))
            if detection_query.first() is None:
                self.logger.debug("All selected detections already annotated, skipping annotation")
                return

        self.logger.debug("Fetching and processing detections from database")

        self.generate_descriptors(detection_query, page_size, process_pool, force_update)

        self.logger.debug("Committing annotations into database")
        connection.session.commit()
        self.logger.info("Annotations finished and committed")

    @abc.abstractmethod
    def generate_descriptors(self, detection_query: sqlalchemy.orm.Query, page_size: int,
                             process_pool: multiprocessing.Pool, force_update: bool) -> None:
        '''
        A method that processes a query of detections and writes the feature vectors into the database
        '''
        pass


class MultiJobAnnotator(Annotator, ABC):
    '''
    Base class for the annotators that supports multiple threads (subprocesses)
    '''
    class SingleProcessPool:
        @staticmethod
        def map(func, iterable, chunksize=None):
            return [func(item) for item in iterable]

        @staticmethod
        def starmap(func, iterable, chinksize=None):
            return [func(*params) for params in iterable]

        @staticmethod
        def close():
            pass

        @staticmethod
        def __enter__(self):
            return self

        @staticmethod
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def __init__(self, *args, **kwargs):
        self.process_pool = None
        super(MultiJobAnnotator, self).__init__(*args, **kwargs)

    def annotate(self, camera_ids: typing.List[int], class_: typing.Optional[str], page_size: int = 5000, threads: int = 1,
                 force_update: bool = False, end: typing.Optional[datetime.timedelta] = None,
                 process_pool: typing.Optional[multiprocessing.Pool] = None,
                 by_identity_model: typing.Optional[int] = None) -> None:
        if process_pool is not None:
            return super().annotate(camera_ids, class_, page_size, threads, force_update, end, process_pool, by_identity_model)
        if threads > 1:
            with multiprocessing.Pool(threads) as process_pool:
                return super().annotate(camera_ids, class_, page_size, threads, force_update, end, process_pool, by_identity_model)
        return super().annotate(camera_ids, class_, page_size, threads, force_update, end, self.SingleProcessPool(), by_identity_model)


class StandardMultiJobAnnotator(MultiJobAnnotator):
    '''Base class that allows for implementation of method annotate_single_detection_static. Results of this method
    will be processed in multithread fashion'''
    def annotate_single_detection(self, detection: connection.Detection) -> typing.Optional[bytes]:
        return self.annotate_single_detection_static(self.parameters, detection)

    @staticmethod
    @abc.abstractmethod
    def annotate_single_detection_static(parameters: dict, detection: connection.Detection) -> typing.Optional[bytes]:
        pass

    def generate_descriptors(self, detection_query: sqlalchemy.orm.Query, page_size: int,
                             process_pool: multiprocessing.Pool, force_update: bool) -> None:
        this_annotate_single_detection = functools.partial(self.annotate_single_detection_static, self.parameters)
        this_descriptor_generator = functools.partial(descriptor_generator,
                                                      self.feature_type.id, this_annotate_single_detection)

        if page_size > 0:
            if force_update:
                raise RuntimeError("Force update with paging is not supported in this annotator")
            paged_query = utils.smooth_paged_query(detection_query, page_size)
            detection_iterator = tqdm.tqdm(paged_query, desc="Annotating detections", total=detection_query.count())
        else:
            detection_iterator = tqdm.tqdm(detection_query, desc="Annotating detections", total=detection_query.count())

        descriptors = process_pool.starmap(this_descriptor_generator, detection_iterator)
        connection.session.bulk_save_objects(filter(lambda desc: desc is not None, descriptors))


class DummyAnnotator(StandardMultiJobAnnotator):
    '''Dummy annotator that annotates with empty feature vector'''
    is_public = True

    @staticmethod
    def annotate_single_detection_static(parameters: dict, detection: connection.Detection) -> typing.Optional[bytes]:
        return b''


class HistogramAnnotator(StandardMultiJobAnnotator):
    '''Base class for all histogram approaches'''
    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser(argument_parser)
        parser.add_argument('-b', '--bins', action='store', type=int, required=True,
                            help='Number of bins of histogram')
        return parser

    @classmethod
    @abc.abstractmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        pass

    @classmethod
    @abc.abstractmethod
    def generate_histogram(cls, data: np.ndarray, bins: int, **kwargs) -> np.ndarray:
        pass

    @classmethod
    @abc.abstractmethod
    def annotate_single_detection_static(cls, parameters: dict, detection: connection.Detection) -> typing.Optional[bytes]:
        pass


class SimpleHistogramAnnotator(HistogramAnnotator):
    '''Base class for histogram approaches without background filtering'''

    @classmethod
    def annotate_single_detection_static(cls, parameters: dict, detection: connection.Detection) -> typing.Optional[bytes]:
        data = cls.prepare_image_data(detection.image, parameters)
        histogram = np.array(cls.generate_histogram(data, parameters['bins']), dtype=np.float32)
        norm_histogram = histogram.reshape(-1) / np.sum(histogram)
        return bytes(norm_histogram)


class CroppedHistogramAnnotator(HistogramAnnotator):
    '''Base class for histogram approaches with background filtering by cropping'''
    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)

        if self.argument_parser is not None:
            error_messenger = self.argument_parser.error
        else:
            def error_messenger(msg: str):
                raise RuntimeError(msg)

        if self.parameters['x_axis'][0] < 0:
            error_messenger("Left boundary must be at least 0")
        if self.parameters['x_axis'][1] > 1:
            error_messenger("Right boundary must be at most 1")
        if self.parameters['x_axis'][1] <= self.parameters['x_axis'][0]:
            error_messenger("Left boundary must be less than right boundary")
        if self.parameters['y_axis'][0] < 0:
            error_messenger("Top boundary must be at least 0")
        if self.parameters['y_axis'][1] > 1:
            error_messenger("Bottom boundary must be at most 1")
        if self.parameters['y_axis'][1] <= self.parameters['y_axis'][0]:
            error_messenger("Top boundary must be less than bottom boundary")

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser(argument_parser)
        parser.add_argument('-x', '--x-axis', type=float, nargs=2, default=[0, 1],
                            help='Boundaries of the crop on x-axis, relative to the width of the detection')
        parser.add_argument('-y', '--y-axis', type=float, nargs=2, default=[0, 1],
                            help='Boundaries of the crop on y-axis, relative to the height of the detection')
        return parser

    @classmethod
    def annotate_single_detection_static(cls, parameters: dict, detection: connection.Detection) -> bytes:
        left_pixel = round(parameters['x_axis'][0] * detection.width)
        right_pixel = round(parameters['x_axis'][1] * detection.width)
        top_pixel = round(parameters['y_axis'][0] * detection.height)
        bottom_pixel = round(parameters['y_axis'][1] * detection.height)

        data = cls.prepare_image_data(detection.image[top_pixel:bottom_pixel, left_pixel:right_pixel], parameters)
        histogram = np.array(cls.generate_histogram(data, parameters['bins']), dtype=np.float32)
        norm_histogram = histogram.reshape(-1) / np.sum(histogram)
        return bytes(norm_histogram)


class GaussHistogramAnnotator(HistogramAnnotator):
    '''Base class for histogram with Gaussian weighting'''
    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser(argument_parser)
        parser.add_argument('-x', '--loc-x', action='store', type=float,
                            help='Mean of the gaussian weight on x-axis; relative to the width of the crop')
        parser.add_argument('-y', '--loc-y', action='store', type=float,
                            help='Mean of the gaussian weight on y-axis; relative to the height of the crop')
        parser.add_argument('-s', '--scale', action='store', type=float,
                            help='Standard deviation of the gaussian weight; relative to respective dimensions of the'
                                 ' crop')
        return parser

    @classmethod
    def annotate_single_detection_static(cls, parameters: dict, detection: connection.Detection) -> bytes:
        data = cls.prepare_image_data(detection.image, parameters)
        linspace_x = np.linspace(0, 1, detection.width)
        weights_x = scipy.stats.norm.pdf(linspace_x, loc=parameters['loc_x'], scale=parameters['scale'])
        linspace_y = np.linspace(0, 1, detection.height)
        weights_y = scipy.stats.norm.pdf(linspace_y, loc=parameters['loc_y'], scale=parameters['scale'])
        weights = weights_y[:, np.newaxis] * weights_x
        flatten_weights = weights.reshape(-1)

        histogram = np.array(cls.generate_histogram(data, parameters['bins'], weights=flatten_weights), dtype=np.float32)
        norm_histogram = histogram.reshape(-1) / np.sum(histogram)
        return bytes(norm_histogram)


class Histogram1D(HistogramAnnotator):
    '''Base class for histogram annotators with single channel'''

    @classmethod
    def generate_histogram(cls, data: np.ndarray, bins: int, **kwargs) -> np.ndarray:
        histogram, bins = np.histogram(data, bins=bins, range=cls.histogram_range, **kwargs)
        return histogram


class Histogram2D(HistogramAnnotator):
    """Base class for histogram annotators with multiple channels"""

    @classmethod
    def generate_histogram(cls, data: np.ndarray, bins: int, **kwargs) -> np.ndarray:
        histogram, bins = np.histogramdd(data, bins=bins, range=data.shape[-1] * [cls.histogram_range], **kwargs)
        return histogram


class RgbHistogram(Histogram2D):
    """Base class for RGB histogram annotators"""
    histogram_range = (0, 256)

    @classmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        return image.reshape(-1, 3)


class HueHistogram(Histogram1D):
    """Base class for Hue histogram annotators"""
    histogram_range = (0, 1)

    @classmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        return matplotlib.colors.rgb_to_hsv(image / 256)[:, :, 0].reshape(-1)


class BlackWhiteHueHistogram(HistogramAnnotator):
    """Base class for Hue histogram with black and white bins"""
    @classmethod
    def generate_histogram(cls, data: np.ndarray, bins: int, **kwargs) -> np.ndarray:
        this_range = (-1 / bins, 1 + 1 / bins)
        histogram, bins = np.histogram(data, bins=bins + 2, range=this_range, **kwargs)
        return histogram

    @classmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        hsv = matplotlib.colors.rgb_to_hsv(image / 256)
        hue = hsv[:, :, 0].reshape(-1)
        sat = hsv[:, :, 1].reshape(-1)
        val = hsv[:, :, 2].reshape(-1)

        hue[sat < 0.2] = 1.001  # White
        hue[val < 0.2] = -0.001  # Black
        return hue


class BlackWhiteHueSaturationHistogram(HistogramAnnotator):
    """Base class for Hue and Saturation histogram with black and white bins"""
    @classmethod
    def generate_histogram(cls, data: np.ndarray, bins: int, **kwargs) -> np.ndarray:
        this_range = (-1 / bins, 1)
        histogram, bins = np.histogramdd(data, bins=bins+1, range=(this_range, this_range), **kwargs)
        return histogram

    @classmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        hsv = matplotlib.colors.rgb_to_hsv(image / 256)
        sat = hsv[:, :, 1].reshape(-1)
        val = hsv[:, :, 2].reshape(-1)
        hs = hsv[:, :, 0:2].reshape(-1, 2)
        hs[sat < 0.2] = [-0.001, 0.2]  # White
        hs[val < 0.2] = [-0.001, 0.7]  # Black
        return hs


class HueSaturationHistogram(Histogram2D):
    '''Base class for Hue and Saturation histogram'''
    histogram_range = (0, 1)

    @classmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        return matplotlib.colors.rgb_to_hsv(image / 256)[:, :, 0:2].reshape(-1, 2)


class UvHistogram(Histogram2D):
    """Base class for UV histograms"""
    histogram_range = (-1 << 15, 1 << 15)

    convertor = np.array([
        [-43, -84, 127],
        [127, -106, -21],
    ]).T

    @classmethod
    def prepare_image_data(cls, image: np.ndarray, parameters: dict) -> np.ndarray:
        return (image @ cls.convertor).reshape(-1, 2)


class SimpleRgbHistogram(RgbHistogram, SimpleHistogramAnnotator):
    is_public = True


class SimpleHueHistogram(HueHistogram, SimpleHistogramAnnotator):
    is_public = True


class SimpleBlackWhiteHueHistogram(BlackWhiteHueHistogram, SimpleHistogramAnnotator):
    is_public = True


class SimpleHueSaturationHistogram(HueSaturationHistogram, SimpleHistogramAnnotator):
    is_public = True


class SimpleUvHistogram(UvHistogram, SimpleHistogramAnnotator):
    is_public = True


class CroppedHueSaturationHistogram(HueSaturationHistogram, CroppedHistogramAnnotator):
    is_public = True


class CroppedHueHistogram(HueHistogram, CroppedHistogramAnnotator):
    is_public = True


class CroppedRgbHistogram(RgbHistogram, CroppedHistogramAnnotator):
    is_public = True


class CroppedUvHistogram(UvHistogram, CroppedHistogramAnnotator):
    is_public = True


class GaussHueSaturationHistogram(HueSaturationHistogram, GaussHistogramAnnotator):
    is_public = True


class CroppedBlackWhiteHueHistogram(BlackWhiteHueHistogram, CroppedHistogramAnnotator):
    is_public = True


class CroppedBlackWhiteHueSaturationHistogram(BlackWhiteHueSaturationHistogram, CroppedHistogramAnnotator):
    is_public = True


class NeuralNetAnnotator(MultiJobAnnotator):
    """Base class for all the annotators based on neural networks"""
    is_public: bool = False

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser(argument_parser)
        parser.add_argument('-s', '--shape', type=int, required=True, nargs=2,
                            help="Rescale images into this shape prior processing by resnet")
        return parser

    @staticmethod
    def get_reshaped_image_from_detection(shape: typing.Tuple[int, int], detection: connection.Detection) -> np.array:
        return cv2.resize(detection.image, dsize=shape, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def generate_descriptors_inner(feature_type_id, detection, value, feature_descriptor):
        if feature_descriptor is None:
            feature_descriptor = connection.FeatureDescriptor(
                feature_type_id=feature_type_id,
                detection_id=detection.id,
                value=value,
            )
        else:
            feature_descriptor.value = value
        return feature_descriptor

    def generate_descriptors_subquery(self, detection_query: sqlalchemy.orm.Query, process_pool: multiprocessing.Pool,
                                      progress_bar: typing.Optional[tqdm.tqdm] = None) -> None:
        detections_and_feature_descriptors: typing.List[typing.Tuple[connection.Detection, connection.FeatureDescriptor]] = detection_query.all()
        detections, feature_descriptors = zip(*detections_and_feature_descriptors)

        reshape_image = functools.partial(self.get_reshaped_image_from_detection, tuple(self.parameters['shape']))
        reshaped_images = process_pool.map(reshape_image, detections)
        preprocessed_images = self.preprocess_images(np.array(reshaped_images))
        predicted = self.neural_net.predict(preprocessed_images)
        this_generate_descriptors_inner = functools.partial(self.generate_descriptors_inner, self.feature_type.id)

        result = process_pool.starmap(this_generate_descriptors_inner, zip(detections, predicted, feature_descriptors))
        if progress_bar is not None:
            progress_bar.update(len(detections_and_feature_descriptors))

        connection.session.bulk_save_objects(result)
        connection.session.flush()

    def generate_descriptors(self, detection_query: sqlalchemy.orm.Query, page_size: int,
                             process_pool: multiprocessing.Pool, force_update: bool) -> None:
        total = detection_query.count()
        progress_bar = tqdm.tqdm(desc="Annotating detections", total=total)
        if page_size > 0:
            if force_update:
                for subquery in utils.paged_query(detection_query, page_size):
                    self.generate_descriptors_subquery(subquery, process_pool, progress_bar)
            else:
                while detection_query.first():
                    subquery = detection_query.limit(page_size)
                    self.generate_descriptors_subquery(subquery, process_pool, progress_bar)
        else:
            self.generate_descriptors_subquery(detection_query, process_pool)
            progress_bar.update(total)


class PretrainedNeuralNetAnnotator(NeuralNetAnnotator):
    """Base class for annotators without further training"""
    is_public: bool = False

    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.neural_net = self.__class__.neural_net_class(
            weights='imagenet',
            pooling='avg',
            include_top=False,
            input_shape=self.parameters['shape'] + [3]
        )
        self.preprocess_images = self.__class__.neural_net_preprocess_input


class ResnetAnnotator(PretrainedNeuralNetAnnotator):
    """Annotator with base ResNet without training"""
    is_public: bool = True
    neural_net_class = tensorflow.keras.applications.resnet_v2.ResNet50V2
    neural_net_preprocess_input = tensorflow.keras.applications.resnet_v2.preprocess_input


class MobilenetAnnotator(PretrainedNeuralNetAnnotator):
    """Annotator with base MobileNet without training"""
    is_public: bool = True
    neural_net_class = tensorflow.keras.applications.mobilenet_v2.MobileNetV2
    neural_net_preprocess_input = tensorflow.keras.applications.mobilenet_v2.preprocess_input


class ExternalNeuralNetAnnotator(NeuralNetAnnotator):
    """Annotator that annotates with externally supplied neural network (not learned from the data from database)"""
    is_public: bool = True

    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.neural_net = tensorflow.keras.models.load_model(self.parameters['model'])
        self.preprocess_images = getattr(tensorflow.keras.applications, self.parameters['base']).preprocess_input

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser(argument_parser)
        parser.add_argument('-b', '--base',
                            help="Base tensorflow model to use for preprocessing images")
        parser.add_argument('-m', '--model',
                            help="Path to the model to use for annotation")
        return parser


def manhattan_distance_matrix(A):
    '''
    Auxiliary function for training with manhattan distace, creates matrix of all distances between each pair of vectors
    '''
    return tensorflow.math.reduce_sum(tensorflow.math.abs(A[:,:,None] - tensorflow.transpose(A[:,:,None])), axis=1)


class CustomNeuralNetAnnotator(NeuralNetAnnotator):
    """Base class for all the network approaches using neural networks with training"""
    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        architecture_string = repr(self)
        for c in '[] ':
            architecture_string = architecture_string.replace(c, '_')
        if len(architecture_string) > 240:
            logger.debug("Shortening too long architecture string {!r}".format(architecture_string))
            architecture_string = hashlib.sha256(bytes(architecture_string, 'ascii')).hexdigest()
        neural_net_filename = os.path.join(os.path.dirname(__file__), 'neural_nets', architecture_string)
        if self.parameters['checkpoint'] is None:
            logger.debug("Established path to the neural net: {!r}".format(neural_net_filename))
        self.preprocess_images = self.get_image_preprocessor()

        if self.parameters['checkpoint'] is not None:
            original_parameters = dict(self.parameters)
            del original_parameters['checkpoint']
            original_option_string = custom_argparse.argument_string(self.argument_parser, original_parameters)
            original_architecture_string = "{}[{}]".format(self.__class__.__name__, original_option_string)
            for c in '[] ':
                original_architecture_string = original_architecture_string.replace(c, '_')
            if len(original_architecture_string) > 240:
                logger.debug("Shortening too long architecture string {!r}".format(original_architecture_string))
                original_architecture_string = hashlib.sha256(bytes(original_architecture_string, 'ascii')).hexdigest()
            checkpoint_filename = os.path.join(os.path.dirname(__file__), 'neural_nets_checkpoints',
                                               original_architecture_string + '__{epoch:03d}.h5'.format(epoch=self.parameters['checkpoint']))
            logger.debug("Established path to the neural net: {!r}".format(checkpoint_filename))
            self.neural_net = tensorflow.keras.models.load_model(checkpoint_filename)
        elif os.path.exists(neural_net_filename):
            logger.debug("Found trained neural net, using it")
            self.neural_net = tensorflow.keras.models.load_model(neural_net_filename)
        else:
            logger.debug("No trained neural net with given parameter found, training new one")
            log_filename = os.path.join(os.path.dirname(__file__), 'neural_nets_logs', architecture_string)
            checkpoint_filename = os.path.join(os.path.dirname(__file__), 'neural_nets_checkpoints', architecture_string + '__{epoch:03d}.h5')
            self.neural_net = self.train_neural_net(log_filename, checkpoint_filename)
            self.neural_net.save(neural_net_filename)

    @abc.abstractmethod
    def get_model_template(self) -> tensorflow.keras.Model:
        pass

    def train_neural_net(self, log_filename: str, checkpoint_filename) -> tensorflow.keras.Model:
        resizing_function = functools.partial(cv2.resize, dsize=tuple(self.parameters['shape']), interpolation=cv2.INTER_CUBIC)
        preprocessing_function = lambda image: self.preprocess_images(resizing_function(image))

        sequence = TripletLossTrainingSequence(
            ground_truth_trajectory_id=self.parameters['ground_truth'],
            camera_id=self.parameters['train_camera'],
            training_start_offset=None if self.parameters['train_start_time'] is None else utils.parse_timedelta(self.parameters['train_start_time']),
            training_end_offset=None if self.parameters['train_end_time'] is None else utils.parse_timedelta(self.parameters['train_end_time']),
            class_=self.parameters['train_class'],
            preprocessing_function=preprocessing_function,
            minimum_trajectory_length=self.parameters['train_minimum_trajectory_length'],
            seed=self.parameters['train_seed'],
            batch_size=self.parameters['batch_size'],
            interwoven_trajectories=self.parameters['train_interwoven_trajectories'],
            maximum_increase_interwoven_trajectories=self.parameters['train_maximum_increase_interwoven_trajectories'],
            shuffle_batches=self.parameters['train_steps_per_epoch'] is not None,
        )

        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_filename, histogram_freq=1)
        checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filename)

        neural_net = self.get_model_template()
        loss = tensorflow_addons.losses.TripletSemiHardLoss if self.parameters['loss'] == 'semihard'\
            else tensorflow_addons.losses.TripletHardLoss
        distance_func = {
            'euclidean': 'L2',
            'cosine': 'angular',
            'manhattan': manhattan_distance_matrix,
        }[self.parameters['distance']]
        neural_net.compile(
            optimizer=tensorflow.keras.optimizers.Adam(self.parameters['learning_rate']),
            loss=loss(distance_metric=distance_func, margin=self.parameters['train_margin']),
        )

        neural_net.fit(sequence, epochs=self.parameters['epochs'], callbacks=[tensorboard_callback, checkpoint_callback], steps_per_epoch=self.parameters['train_steps_per_epoch'])
        return neural_net

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser()
        parser.add_argument('--train-camera', type=int, required=True,
                            help="ID of camera to train on")
        parser.add_argument('--train-start-time', default=None,
                            help="Time offset of the first frame to start training on")
        parser.add_argument('--train-end-time', default=None,
                            help="Time offset of the first frame to not to train on")
        parser.add_argument('-b', '--batch-size', type=int, default=32,)
        parser.add_argument('-g', '--ground-truth', type=int, required=True,
                            help="ID of trajectory to consider as ground truth")
        parser.add_argument('--train-class', default=None,
                            help="Specify a class to train on")
        parser.add_argument('--train-minimum-trajectory-length', '--traj-len', default=4,
                            help="Specify minimal length of a trajectory for the trajectory to be used during training")
        parser.add_argument('--train-seed', type=int, default=42)
        parser.add_argument('--train-interwoven-trajectories', '--interwoven', type=int, default=4,
                            help="Number of different trajectories in a batch during training")
        parser.add_argument('--train-maximum-increase-interwoven-trajectories', '--interwoven-inc', type=int, default=2,
                            help="This allows to add additional trajectories to a batch to increase variety during training")
        parser.add_argument('-l', '--loss', choices=['hard', 'semihard'], default='hard')
        parser.add_argument('-e', '--epochs', type=int, default=50)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--checkpoint', type=int, default=None,
                            help="Use given checkpoint (epoch) instead of fully trained net, must be already created")
        parser.add_argument('--distance', choices=['euclidean', 'manhattan', 'cosine'], default='euclidean',
                            help="Distance function used for training")
        parser.add_argument('--train-steps-per-epoch', type=int, default=None)
        parser.add_argument('--train-margin', type=float, default=1.0)
        return parser


class AlteredNetAnnotator(CustomNeuralNetAnnotator):
    """Base function for annotators that takes neural network from different approaches, alters it and continue with
    training"""
    is_public = True

    @classmethod
    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser(argument_parser)
        base_annotators = [k for k, v in Annotator.public_annotators.items() if issubclass(v, CustomNeuralNetAnnotator)]
        parser.add_argument('--prepare', '--prep', default=None,
                            help='Base tensorflow model to use for preprocessing images')
        parser.add_argument('--dense', type=int,
                            help="Add dense layer with neurons")
        parser.add_argument('--residual', action='store_true',
                            help='Add residual connections')
        parser.add_argument('--frozen', action='store_true',
                            help="Freeze the base model")
        parser.add_argument('--activation',
                            help="activation function of final layer")
        parser.add_argument('base_annotator', choices=base_annotators,
                            help="Annotator to use for the annotation")
        parser.add_argument('base_annotator_arguments', nargs=argparse.REMAINDER,
                                     help="Additional arguments for chosen annotator")
        return parser

    def get_image_preprocessor(self) -> typing.Callable[[np.array], np.array]:
        if self.parameters['prepare'] is not None:
            return getattr(tensorflow.keras.applications, self.parameters['prepare']).preprocess_input
        else:
            return lambda x: x

    def get_model_template(self) -> tensorflow.keras.Model:
        base_annotator_class = Annotator.public_annotators[self.parameters['base_annotator']]
        base_annotator = base_annotator_class(logger=self.logger, options=self.parameters['base_annotator_arguments'])

        pretrained_model = base_annotator.neural_net

        if self.parameters['frozen']:
            pretrained_model.trainable = False

        if self.parameters['dense'] is not None:
            layer = tensorflow.keras.layers.Dense(self.parameters['dense'], activation=self.parameters['activation'], name="altered_dense")(pretrained_model.output)
        else:
            layer = pretrained_model.output

        if self.parameters['residual']:
            layer = tensorflow.keras.layers.Add()([pretrained_model.output, layer])

        normalize = tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1), name="altered_lambda")(layer)
        model = tensorflow.keras.Model(inputs=pretrained_model.input, outputs=normalize)
        return model


class StandardNeuralNetAnnotator(CustomNeuralNetAnnotator):
    """Annotator with neural net with simple architecture"""
    is_public: bool = True

    def get_image_preprocessor(self) -> typing.Callable[[np.array], np.array]:
        if self.parameters['prepare'] is not None:
            return getattr(tensorflow.keras.applications, self.parameters['prepare']).preprocess_input
        else:
            return lambda x: x

    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser()
        parser.add_argument('--prepare', '--prep', default=None,
                            help='Base tensorflow model to use for preprocessing images')
        return parser

    def get_model_template(self) -> tensorflow.keras.Model:
        return tensorflow.keras.Sequential([
            tensorflow.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                           input_shape=self.parameters['shape'] + [3]),
            tensorflow.keras.layers.MaxPooling2D(pool_size=2),
            tensorflow.keras.layers.Dropout(0.3),
            tensorflow.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            tensorflow.keras.layers.MaxPooling2D(pool_size=2),
            tensorflow.keras.layers.Dropout(0.3),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(256, activation=None),
            tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1))
        ])


class TransferredMobilenetAnnotatorWithDense(CustomNeuralNetAnnotator):
    """Annotator based on MobileNet with added dense layer"""
    is_public: bool = True

    def get_image_preprocessor(self) -> typing.Callable[[np.array], np.array]:
        return tensorflow.keras.applications.mobilenet_v2.preprocess_input

    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser()
        parser.add_argument('--dropout', default=0.4, type=float,
                            help="Frequency of dropout ofter pretrained model")
        parser.add_argument('--dense', default=2048, type=int,
                            help="Size of last dense layer")
        parser.add_argument('--frozen', action='store_true',
                            help="Freeze the base model")
        parser.add_argument('--activation',
                            help="activation function of final layer")
        return parser

    def get_model_template(self) -> tensorflow.keras.Model:
        pretrained_model = tensorflow.keras.applications.MobileNetV2(
            weights='imagenet',
            pooling='avg',
            include_top=False,
            input_shape=self.parameters['shape'] + [3],
        )
        if self.parameters['frozen']:
            pretrained_model.trainable = False

        if self.parameters['dropout'] > 0.00001:
            dropout = tensorflow.keras.layers.Dropout(self.parameters['dropout'])(pretrained_model.output)
        else:
            dropout = pretrained_model.output
        dense = tensorflow.keras.layers.Dense(self.parameters['dense'], activation=self.parameters['activation'])(dropout)
        normalize = tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1))(dense)
        model = tensorflow.keras.Model(inputs=pretrained_model.input, outputs=normalize)
        return model


class TransferredResnetAnnotatorWithDense(CustomNeuralNetAnnotator):
    """Annotator based on ResNet with added dense layer"""
    is_public: bool = True

    def get_image_preprocessor(self) -> typing.Callable[[np.array], np.array]:
        return tensorflow.keras.applications.resnet_v2.preprocess_input

    def setup_argument_parser(cls, argument_parser: typing.Optional[argparse.ArgumentParser] = None) -> \
            argparse.ArgumentParser:
        parser = super().setup_argument_parser()
        parser.add_argument('--dropout', default=0.4, type=float,
                            help="Frequency of dropout ofter pretrained model")
        parser.add_argument('--dense', default=2048, type=int,
                            help="Size of last dense layer")
        parser.add_argument('--frozen', action='store_true',
                            help="Freeze the base model")
        parser.add_argument('--activation',
                            help="activation function of final layer")
        return parser

    def get_model_template(self) -> tensorflow.keras.Model:
        pretrained_model = tensorflow.keras.applications.ResNet50V2(
            weights='imagenet',
            pooling='avg',
            include_top=False,
            input_shape=self.parameters['shape'] + [3],
        )
        if self.parameters['frozen']:
            pretrained_model.trainable = False

        if self.parameters['dropout'] > 0.0000001:
            dropout = tensorflow.keras.layers.Dropout(self.parameters['dropout'], activation=self.parameters['activation'])(pretrained_model.output)
        dense = tensorflow.keras.layers.Dense(self.parameters['dense'])(dropout)
        normalize = tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1))(dense)
        model = tensorflow.keras.Model(inputs=pretrained_model.input, outputs=normalize)
        return model


class TransferredMobilenetAnnotator(CustomNeuralNetAnnotator):
    """Annotator based on MobileNet without the last layer"""
    is_public: bool = True

    def get_image_preprocessor(self) -> typing.Callable[[np.array], np.array]:
        return tensorflow.keras.applications.mobilenet_v2.preprocess_input

    def get_model_template(self) -> tensorflow.keras.Model:
        pretrained_model = tensorflow.keras.applications.MobileNetV2(
            weights='imagenet',
            pooling='avg',
            include_top=False,
            input_shape=self.parameters['shape'] + [3],
        )

        normalize = tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1))(pretrained_model.output)
        model = tensorflow.keras.Model(inputs=pretrained_model.input, outputs=normalize)
        return model


class TransferredResnetAnnotator(CustomNeuralNetAnnotator):
    """Annotator based on ResNet without the last layer"""
    is_public: bool = True

    def get_image_preprocessor(self) -> typing.Callable[[np.array], np.array]:
        return tensorflow.keras.applications.resnet_v2.preprocess_input

    def get_model_template(self) -> tensorflow.keras.Model:
        pretrained_model = tensorflow.keras.applications.ResNet50V2(
            weights='imagenet',
            pooling='avg',
            include_top=False,
            input_shape=self.parameters['shape'] + [3],
        )

        normalize = tensorflow.keras.layers.Lambda(lambda x: tensorflow.math.l2_normalize(x, axis=1))(pretrained_model.output)
        model = tensorflow.keras.Model(inputs=pretrained_model.input, outputs=normalize)
        return model


def main(argument_string: typing.Optional[str] = None):
    argument_parser = argparse.ArgumentParser()
    custom_argparse.set_logger(argument_parser)

    argument_parser.add_argument('-c', '--camera', required=True, nargs="+",
                                 help='ID of cameras to annotate')
    argument_parser.add_argument('--class', type=str, dest='class_', metavar='CLASS',
                                 help='Class to annotate (e.g. PERSON), if omitted all classes are annotated')
    argument_parser.add_argument('-p', '--page-size', type=int, default=5000,
                                 help='Number of detection requested from database at one time. Setting this to low '
                                      'number decreases local RAM usage but increases the number of database requests. '
                                      'To disable paging altogether, set this to 0')
    argument_parser.add_argument('-j', '--jobs', type=int, default=1,
                                 help='Number of jobs (threads) to use')
    argument_parser.add_argument('-f', '--force-update', action='store_true',
                                 help='Recompute descriptors even if they are already present')
    argument_parser.add_argument('--by-identity-model', type=int, default=None)
    argument_parser.add_argument('--end',
                                 help="The time offset of last frame to annotate")
    argument_parser.add_argument('annotator', choices=Annotator.public_annotators.keys(),
                                 help="Annotator to use for the annotation")
    argument_parser.add_argument('annotator_arguments', nargs=argparse.REMAINDER,
                                 help="Additional arguments for chosen annotator")

    parsed_arguments = argument_parser.parse_args(argument_string)
    AnnotatorClass: typing.Type['Annotator'] = Annotator.public_annotators[parsed_arguments.annotator]
    logger = utils.get_logger('Annotator', parsed_arguments.log)
    logger.debug("Parameters processed")
    annotator: Annotator = AnnotatorClass(logger=logger, options=parsed_arguments.annotator_arguments)
    end = None if parsed_arguments.end is None else utils.parse_timedelta(parsed_arguments.end)
    annotator.annotate(
        camera_ids=parsed_arguments.camera,
        class_=parsed_arguments.class_,
        page_size=parsed_arguments.page_size,
        threads=parsed_arguments.jobs,
        force_update=parsed_arguments.force_update,
        end=end,
        by_identity_model=parsed_arguments.by_identity_model,
    )


if __name__ == '__main__':
    main()
