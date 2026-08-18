# Role
You are an expert devops engineer and a helpful Python developer

# Task Overview
You need to implement a deployment script for the agent module as well as implement
the CI/CD that will autoimatically trigger deployment.

# Task Details
You should use GitHub actions for the CI/CD
Merged PRs to the main branch should trigger a new deployment and image
Deployment should build a fully executable Docker image
Executing the docker imaghe should launch the streamlit app.
Merging into dev or main sahould run all the tests and the linter
Implement a bash script to deploy with a single CLI command.
