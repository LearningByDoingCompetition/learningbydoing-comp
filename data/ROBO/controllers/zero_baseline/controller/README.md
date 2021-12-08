# Submission template

Start implementing your controller in controller.py 😎

For your very first submission, you may just upload a zip file containing
* controller.py
* metadata
* start.sh
* wrapcontroller.py

Only change start.sh and wrapcontroller.py,
if you know you have to and what you are doing! ⚠️

## Using another docker image

Per default, your controller is run within the docker image
docker.io/learningbydoingdocker/codalab-competition-docker:latest
as specified in the default metadata file.
If you need other packages or wish to use another environment,
you can specify another docker image to use in the metadata file.

If your controller is python based, just keep on using wrapcontroller.py
and start.sh from this submission template, and make sure the selected
docker image has pyzmq installed.

Note: The containers do not have internet access and restricted resources
during evaluation.

### Advanced

If you intend on implementing your controller in R, Julia, C, ...
it may be the easiest to rely on the provided IPC interface and adapt
wrapcontroller.py or controller.py to call your controller implementation.

### More advanced

If you cannot build up on the provided IPC interface (that is, pyzmq and the
provided start.sh and wrapcontroller.py), you may implement an interface that
listens on ipc://socket analogous to that in wrapcontroller.py.

