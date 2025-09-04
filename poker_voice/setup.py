from setuptools import find_packages, setup

package_name = 'poker_voice'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['resource/models/hello_rokey_8332_32.tflite']),
        ('share/' + package_name + '/audio', ['resource/audio/또깡.mp3', 'resource/audio/알았음.mp3', 'resource/audio/에러.mp3']),
    ],
    install_requires=['setuptools', 'openai', 'sounddevice', 'scipy', 'numpy', 'openwakeword', 'pyaudio', 'pydub', 'langchain', 'langchain-openai'],
    zip_safe=True,
    maintainer='happy',
    maintainer_email='happy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_recognizer_node = poker_voice.voice_recognizer_node:main',
        ],
    },
)
