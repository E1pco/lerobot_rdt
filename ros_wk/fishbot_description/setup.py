from setuptools import setup
from glob import glob
import os

package_name = 'fishbot_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/**')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/**')),
        (os.path.join('share', package_name, 'config'), glob('config/**')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='SO-101 Robot Description and IK Solver',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_solver_node = fishbot_description.ik_solver_node:main',
            'target_pose_publisher = fishbot_description.target_pose_publisher:main',
            'target_pose_input = fishbot_description.target_pose_input:main',
        ],
    },
)
