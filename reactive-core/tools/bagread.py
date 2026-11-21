"""Read-only rosbag2 (sqlite3) helper for the offline tools. Needs ROS 2 Humble sourced."""
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read(bag_dir, topics=None):
    """Yield (topic, msg, t_ns) in recording order. t_ns = receive time in the bag."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    cls = {}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in cls:
            cls[topic] = get_message(types[topic])
        yield topic, deserialize_message(data, cls[topic]), t


def start_ns(bag_dir):
    info = rosbag2_py.Info().read_metadata(bag_dir, 'sqlite3')
    return info.starting_time.nanoseconds
