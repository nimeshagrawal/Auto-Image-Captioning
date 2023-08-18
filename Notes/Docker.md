# Docker pipeline

1. Within git repo we will be needing a 'Dockerfile' file. It contains all the base-image, and sequential commands to be executed, and the properties of the container when it is built.

## To sync to DockerHub via github actions.
The Github actions is designed in the following manner.
1. it is triggered on an event [push or pull on specific branch/branches]
2. Determine the machines on which the actions must be built and tested.
3. Store the DOCKERHUB_USERNAME and DOCKERHUB_TOKEN as github-secrets.[repo settings->secrets->enter secret varialbles]
4. specify the DockerHub-repo/space to where the current repo contents must be synced.
