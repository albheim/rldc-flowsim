# Reinforcement Learning for Data Centers - Flow simulation and control 
This repository contains files for a simple simulation of the flow and energy usage of a DC expressed as an RL environment.
It also has some code for running a basic RL experiment on this environment.

## Running the simulation
First you need to set up the environment, a suggestion is to use some kind of virtual environment for example conda. We used Miniconda with Python 3.8.5 for our simulations. After setting up the environment you will need to start ray, provided is an example of how to start a local instance of ray. After that is an example of how to start the simulation which will run for 2M steps (each step is one second in the simulated environment).
```
pip install numpy tensorflow ray ray[rllib] ray[tune] matplotlib seaborn pandas
ray start --head 
python src/main.py --stop_iterations 2000000
```

### Stop ray
Ray will run in the background if not stopped which can be done with
```
ray stop
```
### Tensorboard
Data is by default logged to `~/ray_results/` in tensorboard format, and to view it you can run 
```
tensorboard --logdir ~/ray_results
```
after which you navigate to `localhost:6006` (or whatever tensorboard told you) to view the data.

## Python environment
This is the python environment used to run the code. It will likely work with other versions, but is documented for completeness.
```
Package                  Version
------------------------ ---------
absl-py                  0.12.0
aiohttp                  3.7.3
aiohttp-cors             0.7.0
aioredis                 1.3.1
astunparse               1.6.3
async-timeout            3.0.1
atari-py                 0.2.6
attrs                    20.3.0
backcall                 0.2.0
blessings                1.7
brotlipy                 0.7.0
cachetools               4.2.1
certifi                  2020.12.5
cffi                     1.14.3
chardet                  3.0.4
click                    7.1.2
cloudpickle              1.6.0
colorama                 0.4.4
colorful                 0.5.4
conda                    4.9.2
conda-package-handling   1.7.2
cryptography             3.2.1
cycler                   0.10.0
decorator                4.4.2
dm-tree                  0.1.5
filelock                 3.0.12
flatbuffers              1.12
future                   0.18.2
gast                     0.3.3
google-api-core          1.25.1
google-auth              1.27.1
google-auth-oauthlib     0.4.3
google-pasta             0.2.0
googleapis-common-protos 1.52.0
gpustat                  0.6.0
grpcio                   1.32.0
gym                      0.18.0
h5py                     2.10.0
hiredis                  1.1.0
idna                     2.10
ipykernel                5.3.4
ipython                  7.21.0
ipython-genutils         0.2.0
jedi                     0.17.0
jsonschema               3.2.0
jupyter-client           6.1.7
jupyter-core             4.7.1
Keras-Preprocessing      1.1.2
kiwisolver               1.3.1
lz4                      3.1.3
Markdown                 3.3.4
matplotlib               3.3.4
msgpack                  1.0.2
multidict                5.1.0
numpy                    1.19.5
nvidia-ml-py3            7.352.0
oauthlib                 3.1.0
opencensus               0.7.12
opencensus-context       0.1.2
opencv-python            4.5.1.48
opencv-python-headless   4.3.0.36
opt-einsum               3.3.0
pandas                   1.2.1
parso                    0.8.1
pexpect                  4.8.0
pickleshare              0.7.5
Pillow                   7.2.0
pip                      21.0.1
prometheus-client        0.9.0
prompt-toolkit           3.0.17
protobuf                 3.15.6
psutil                   5.8.0
ptyprocess               0.7.0
py-spy                   0.3.4
pyasn1                   0.4.8
pyasn1-modules           0.2.8
pycosat                  0.6.3
pycparser                2.20
pyglet                   1.5.0
Pygments                 2.8.1
pyOpenSSL                19.1.0
pyparsing                2.4.7
pyrsistent               0.17.3
PySocks                  1.7.1
python-dateutil          2.8.1
pytz                     2020.5
PyYAML                   5.4.1
pyzmq                    20.0.0
ray                      1.1.0
redis                    3.5.3
requests                 2.25.1
requests-oauthlib        1.3.0
rsa                      4.7.2
ruamel-yaml              0.15.87
scipy                    1.4.1
seaborn                  0.11.1
setuptools               54.1.2
six                      1.15.0
tabulate                 0.8.7
tensorboard              2.4.1
tensorboard-plugin-wit   1.8.0
tensorboardX             2.1
tensorflow               2.4.1
tensorflow-estimator     2.4.0
termcolor                1.1.0
tornado                  6.1
tqdm                     4.51.0
traitlets                5.0.5
typing-extensions        3.7.4.3
urllib3                  1.26.4
wcwidth                  0.2.5
Werkzeug                 1.0.1
wheel                    0.36.2
wrapt                    1.12.1
yarl                     1.6.3
```
