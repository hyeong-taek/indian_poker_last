from setuptools import find_packages, setup

package_name = 'indian_poker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyeonhee',
    maintainer_email='lhh6225739@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot = indian_poker.poker_robot_ver4:main',
            'dsr_driver = indian_poker.dsr_driver:main',
            'orchestrator = indian_poker.poker_orchestrator_final:main',
            'camer_mock = indian_poker.camera_mock:main'
        ],
    },
)
