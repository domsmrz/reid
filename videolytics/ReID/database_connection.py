"""List of classes for direc communication with database"""


import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import *
from sqlalchemy.ext.declarative import *
from sqlalchemy.ext.hybrid import *
import typing

from . import utils
from . import string_utils
from . import database_settings

engine = create_engine(
    database_settings.database,
    executemany_mode='values',
)
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()


def pascal_to_snake_case(input_string):
    return string_utils.CaseTransformer.from_pascal(input_string).to_snake()


@as_declarative()
class Base:
    @declared_attr
    def __tablename__(cls):
        return pascal_to_snake_case(cls.__name__)

    def __repr__(self):
        str_attributes = ["id={}".format(self.id)]
        str_attributes.extend("{}={!r}".format(k, v) for k, v in self.__dict__.items() if not k.startswith('_') and k != 'id')
        return "{}({})".format(self.__class__.__name__, ', '.join(str_attributes))

    id = Column(Integer, primary_key=True)


def add_simple_1_n_relationship(parent_class: typing.Type[Base], child_class: typing.Type[Base],
                                child_column_name: typing.Optional[str] = None,
                                parent_key: typing.Optional[str] = None,
                                child_key: typing.Optional[str] = None, nullable: bool = False,
                                child_relationship_options: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                parent_relationship_options: typing.Optional[typing.Dict[str, typing.Any]] = None):
    if child_column_name is None:
        child_column_name = pascal_to_snake_case(parent_class.__name__)
    if parent_key is None:
        parent_key = string_utils.pluralize(pascal_to_snake_case(child_class.__name__))
    if child_key is None:
        child_key = pascal_to_snake_case(parent_class.__name__)

    if child_relationship_options is None:
        child_relationship_options = dict()
    if parent_relationship_options is None:
        parent_relationship_options = dict()

    child_column = Column(child_column_name, Integer, ForeignKey(parent_class.id), nullable=nullable)
    setattr(child_class, child_key + '_id', child_column)
    child_relationship = relationship(parent_class, back_populates=parent_key, **child_relationship_options)
    setattr(child_class, child_key, child_relationship)
    parent_relationship = relationship(child_class, back_populates=child_key, **parent_relationship_options)
    setattr(parent_class, parent_key, parent_relationship)


class Camera(Base):
    name = Column(Text, nullable=False)
    ip = Column(Text, nullable=False)
    port = Column(Integer, nullable=False)
    codec = Column(Text, nullable=False)
    fps = Column(Integer, nullable=False)
    h = Column(Integer, nullable=False)
    w = Column(Integer, nullable=False)


class Frame(Base):
    timestamp = Column(DateTime, nullable=False)
    sequence_number = Column(Integer, nullable=False)


add_simple_1_n_relationship(parent_class=Camera, child_class=Frame)


class Detection(Base):
    class_ = Column('class', Text, nullable=False)
    left = Column(Integer, nullable=False)
    top = Column(Integer, nullable=False)
    right = Column(Integer, nullable=False)
    bottom = Column(Integer, nullable=False)
    crop = Column(Binary, nullable=False)
    conf = Column(Float, nullable=False)
    feature = Column(Binary)

    @property
    def dimensions(self):
        return np.array([self.bottom - self.top, self.right - self.left], dtype=np.uint32)

    @hybrid_property
    def width(self):
        return self.right - self.left

    @hybrid_property
    def height(self):
        return self.bottom - self.top

    @property
    def position(self):
        return np.array([self.left, self.top], dtype=np.uint32)

    @property
    def image(self):
        return np.frombuffer(self.crop, dtype=np.uint8).reshape(*self.dimensions, 3)

    @hybrid_property
    def center_x(self):
        return (self.left + self.right) / 2

    @hybrid_property
    def center_y(self):
        return (self.bottom + self.top) / 2

    @property
    def center(self):
        return np.array([self.center_x, self.center_y])


setattr(Detection, 'class', Detection.class_)
add_simple_1_n_relationship(parent_class=Frame, child_class=Detection)


class FeatureType(Base):
    __tablename__ = 'feature_type'
    annotator_name = Column('annotator', String, nullable=False)
    options = Column(String)
    description = Column(Text)

    __table_args__ = (UniqueConstraint(annotator_name, options), )


class FeatureDescriptor(Base):
    __tablename__ = 'feature_descriptor'
    value = Column(Binary, nullable=False)


add_simple_1_n_relationship(parent_class=Detection, child_class=FeatureDescriptor)
add_simple_1_n_relationship(parent_class=FeatureType, child_class=FeatureDescriptor)
FeatureDescriptor.__table_args__ = (
    UniqueConstraint(FeatureDescriptor.detection_id, FeatureDescriptor.feature_type_id),
)


class TrajectoryModel(Base):
    __tablename__ = 'traj_model'

    generator_name = Column('generator', String, nullable=False)
    options = Column(String)
    description = Column(Text)


add_simple_1_n_relationship(parent_class=FeatureType, child_class=TrajectoryModel, nullable=True)
TrajectoryModel.__table_args__ = (
    UniqueConstraint(TrajectoryModel.generator_name, TrajectoryModel.options, TrajectoryModel.feature_type_id),
)


class Trajectory(Base):
    __tablename__ = 'traj'


add_simple_1_n_relationship(parent_class=TrajectoryModel, child_class=Trajectory, child_column_name='traj_model')
add_simple_1_n_relationship(parent_class=Camera, child_class=Trajectory)


class TrajectoryDetection(Base):
    __tablename__ = 'traj_detection'


add_simple_1_n_relationship(parent_class=Trajectory, child_class=TrajectoryDetection, child_column_name='traj')
add_simple_1_n_relationship(parent_class=Detection, child_class=TrajectoryDetection)
TrajectoryDetection.__table_args__ = (
    UniqueConstraint(TrajectoryDetection.trajectory_id, TrajectoryDetection.detection_id),
)


class CameraSetup(Base):
    description = Column(Text)


class CameraSetupAssignment(Base):
    pass


add_simple_1_n_relationship(parent_class=CameraSetup, child_class=CameraSetupAssignment)
add_simple_1_n_relationship(parent_class=Camera, child_class=CameraSetupAssignment)


class IdentityModel(Base):
    generator_name = Column('generator', String, nullable=False)
    options = Column(String)
    description = Column(Text)


add_simple_1_n_relationship(parent_class=FeatureType, child_class=IdentityModel, nullable=True)
add_simple_1_n_relationship(parent_class=TrajectoryModel, child_class=IdentityModel, nullable=True)

IdentityModel.__table_args__ = (
    UniqueConstraint(
        IdentityModel.generator_name,
        IdentityModel.options,
        IdentityModel.feature_type_id,
        IdentityModel.trajectory_model_id,
    )
)


class Identity(Base):
    pass


add_simple_1_n_relationship(parent_class=IdentityModel, child_class=Identity)
add_simple_1_n_relationship(parent_class=CameraSetup, child_class=Identity)


class IdentityDetection(Base):
    pass


add_simple_1_n_relationship(parent_class=Identity, child_class=IdentityDetection)
add_simple_1_n_relationship(parent_class=Detection, child_class=IdentityDetection)

# Tables used in previous versions of the tables, not used anymore
#
# class OldTrajectoryModel(Base):
#     __tablename__ = 'trajs_models'
#
#     id = Column('model_id', Integer, primary_key=True)
#     description = Column(Text, nullable=False)
#
#     feature_type_id = NotImplemented
#     feature_type = NotImplemented
#
#
# class OldTrajectoryDetection(Base):
#     __tablename__ = 'trajs_boxes'
#
#     id = NotImplemented
#     trajectory_id = Column('traj_id', Integer, nullable=False, primary_key=True)
#     model_id = Column('model_id', Integer, nullable=False)
#     detection_id = Column('box_id', Integer, ForeignKey(Detection.id), nullable=False, primary_key=True)
#
#
# class OldTrajectorySmooth(Base):
#     __tablename__ = 'trajs_smooth'
#
#     id = NotImplemented
#     trajectory_id = Column('traj_id', Integer, nullable=False, primary_key=True)
#     model_id = Column('model_id', Integer, nullable=False, primary_key=True)
#     frame_id = Column('frame_id', Integer, nullable=False, primary_key=True)
#     x = Column(Integer, nullable=False)
#     y = Column(Integer, nullable=False)



if __name__ == '__main__':
   Base.metadata.create_all()
