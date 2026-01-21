## CI, CT and CD Plan for the ASL-Translator Project

### Context and goals

For this project we are building the ASL-Translator system, which consists of multiple Dockerized services (`api`, `inference`, `training`) orchestrated via `docker-compose.yaml`. Model training and retraining are already integrated with MLflow (for example in `services/training/src/model_trainer.py` and `services/training/retrain_worker.py`), and we run everything on a single VM that has Docker installed.  
In this section I describe what I researched about Continuous Integration (CI), Continuous Testing (CT) and Continuous Delivery/Deployment (CD), which options I considered, and what I plan to implement for this specific setup.

---

### Continuous Integration (CI)

**Goal (what we want)**  
With CI we want every change to be automatically checked as soon as it is pushed to GitHub or opened as a pull request. The idea is that broken code, missing dependencies or invalid Docker configuration are caught before we deploy anything to the VM.

**Options I considered**

- **No CI / manual checks only**  
  The simplest option would be to keep running everything manually on our laptops or directly on the VM. This is fast to start with, but very risky: one team member can accidentally push code that doesn’t even start, or that breaks the Docker setup, and we would only notice after deployment.
  
- **GitHub Actions-based CI**  
  Since our code is already on GitHub, GitHub Actions is a natural choice. It is integrated, free for this scale, and supports both Python tools and Docker. We can:
  - Automatically check out the repo.
  - Install Python and our dependencies.
  - Run basic checks (syntax, imports).
  - Validate and build our Docker services.

- **External CI (GitLab CI, Jenkins, etc.)**  
  I also briefly looked at other CI systems (GitLab CI, Jenkins, etc.), but they would either require moving the repository or maintaining additional infrastructure. For a single student project this is unnecessary overhead.

**Plan / what I’m going to implement**

We decided to use **GitHub Actions** for CI by adding a workflow file under `.github/workflows/ci.yml`. The core of this CI job will be:

- **Checkout**: Use `actions/checkout@v4` so the workflow runs against the exact commit that was pushed.
- **Python setup**: Use `actions/setup-python@v5` with Python 3.11, which matches the environment we use in our Docker images.
- **Dependency installation**: In the `ASL-Translator` directory run:
  - `pip install -r requirements.txt`  
  to make sure a clean environment can install everything we need.
- **Basic correctness checks**: Run:
  - `python -m compileall .`  
  to catch syntax errors and obvious import problems across all Python modules.
- **Docker validation**: Run from `ASL-Translator`:
  - `docker compose -f docker-compose.yaml config`
  - `docker compose -f docker-compose.yaml build`  
  so we know that `docker-compose.yaml` is valid and that the `api`, `inference` and `training` images all build successfully in CI before we ever try to deploy them.

Later, if we have time, I want to extend this CI job with **static analysis and style checks** (for example `flake8` or `ruff`, maybe `black --check` and `mypy`) to enforce a cleaner code style. For now I focus on getting the core CI pipeline in place and stable.

---

### Continuous Testing (CT)

**Goal (what we want)**  
CT means that our tests are run automatically on every change. The intention is that whenever we refactor or add new features, we immediately get feedback if we broke training, inference or the API, instead of discovering this manually.

**What is testable in this project**

While this is a ML project, a lot of the logic is actually testable without running full heavy trainings:

- **Training logic**  
  In `services/training/src/model_trainer.py` we have the `ModelTrainer` class that handles data preparation, training and evaluation. With small synthetic NumPy arrays we can:
  - Test `prepare_data` (splitting, scaling, label encoding).
  - Test `train_model` (model can be fit without error).
  - Test `evaluate_model` (basic metrics are returned with the correct shape).

- **MLflow integration**  
  `train_with_mlflow` wraps the training with MLflow logging. In tests we can point MLflow to a temporary directory and assert that a run is created and that it logs metrics like accuracy and F1.

- **Retraining logic**  
  `services/training/retrain_worker.py` orchestrates checking for feedback data and triggering a retrain. We can test that, for example, when the feedback threshold is exceeded it calls `ModelTrainer.train_with_mlflow` and updates the database records correctly (possibly with mocks or a temporary DB).

- **Inference and API**  
  In `services/inference/src` and `services/api/src/api.py` we can write:
  - Unit tests for the core inference functions (given a dummy feature vector, they return a prediction in the right format).
  - Lightweight integration tests for the REST API endpoints (using a test client like FastAPI’s `TestClient`).

**Plan / what I’m going to implement**

I plan to introduce a `tests/` directory inside `ASL-Translator` (e.g. `ASL-Translator/tests/test_training.py`, `test_inference.py`, `test_api.py`) and use **pytest** as the main test runner.

In the GitHub Actions workflow I will add a step that:

- Installs `pytest` (if it is not already in `requirements.txt`).
- Checks if a `tests/` directory exists.
- If it exists, runs `pytest` so that any test we add in the future is executed automatically in CI.

In terms of test strategy, I will start with **small unit tests** around `ModelTrainer` and the inference logic, because these are critical for the correctness of the model and relatively easy to test with small, synthetic data. If time allows, I’ll add simple API tests as well, so that at least the main endpoints are covered in CT.

---

### Continuous Delivery / Deployment (CD)

**Goal (what we want)**  
For CD we want new, validated versions of the application to be deployed to our Docker VM with as little manual work as possible, but without adding too much complexity. Every time we push a stable commit to the main branch, the VM should update itself automatically.

**Deployment options I looked at**

- **Manual deployment on the VM**  
  The “old-school” way is to SSH into the VM, run `git pull`, and then `docker compose up -d --build`. This works, but it is easy to forget steps, and it does not scale well for a team.

- **SSH-based deployment from GitHub Actions (no registry)**  
  With this approach the CI job is followed by a deploy job that:
  - SSH-es into the VM.
  - Runs `git pull` in the application directory.
  - Runs `docker compose -f docker-compose.yaml up -d --build`.  
  This reuses the same workflow file and uses GitHub Secrets for the VM credentials. It is simple and fits our current architecture (single VM, Docker Compose).

- **Registry-based deployment (build images in CI, pull images on the VM)**  
  A more advanced option would be:
  - Build Docker images for `api`, `inference` and `training` in CI.
  - Push them to a registry like GitHub Container Registry.
  - On the VM, `docker compose pull` the new images and then `docker compose up -d`.  
  This is cleaner in larger setups and makes rollbacks easier, but it requires extra configuration (registries, credentials, image tags) that might be overkill for this course project.

- **Blue–green or canary deployments**  
  I also briefly looked at patterns like blue–green deployments, where two versions run side by side and we slowly shift traffic. These techniques are common in production, but for a single VM student project they are too complex and not needed.

**Plan / what I’m going to implement**

For this project I plan to implement the **SSH-based deployment from GitHub Actions**:

- In `.github/workflows/ci.yml` I’ll add a second job named `deploy` that:
  - Has `needs: build-and-test`, so it only runs if CI passed.
  - Is restricted to the deployment branch (in our case `refs/heads/master`), so it does not deploy from feature branches or pull requests.
- The `deploy` job will use an SSH action (for example `appleboy/ssh-action`) with the following secrets configured in the GitHub repository:
  - `VM_HOST`: IP or hostname of the VM.
  - `VM_USER`: SSH username.
  - `VM_SSH_KEY`: private SSH key.
  - `VM_SSH_PORT`: SSH port (usually 22).
- On the VM, the deployment script will:
  - `cd /opt/asl-translator` (or wherever the repo is cloned).
  - Run `git pull` to fetch the latest version.
  - Run `docker compose -f docker-compose.yaml up -d --build` to rebuild and restart the services.

This gives us a straightforward CD pipeline:

1. We push code to `master`.
2. CI builds and tests our application and Docker images.
3. If everything passes, GitHub Actions automatically deploys the changes to the VM via SSH.

If we have extra time at the end of the project, a possible improvement would be to switch to the registry-based approach (build images in CI and only pull on the VM), but for now the SSH + `git pull` strategy is the most pragmatic.

---

### MLflow in the CI/CT/CD Story

MLflow is already part of our training and retraining flow:

- `ModelTrainer.train_with_mlflow` sets up a file-based MLflow tracking URI under `/app/models/mlruns`, defines an experiment, logs hyperparameters and metrics, and registers the trained model with `mlflow.sklearn.log_model`.
- The retraining worker (`services/training/retrain_worker.py`) calls `train_with_mlflow` when enough feedback data has been collected, so each retrain is tracked as a separate MLflow run.

In the context of CI/CT/CD, MLflow mainly acts as our **experiment tracking and model provenance tool**:

- CI ensures that the code using MLflow at least builds and runs basic checks.
- CT can include small tests that verify MLflow logging works (by pointing MLflow to a temporary directory in the tests).
- CD ensures that when a new version of the code is deployed on the VM, all subsequent training and retraining runs are automatically logged to MLflow with the new code version.

We could in theory introduce a central MLflow tracking server (for example as another Docker service in `docker-compose.yaml`) and configure the app via environment variables to log there, but given the scope of this project a local file-based MLflow tracking setup is sufficient and much simpler to operate.

---

### Summary of what I plan to build

- **CI**: A GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push/PR, installs dependencies, compiles Python code, and validates/builds the Docker services.
- **CT**: A pytest-based test suite under `ASL-Translator/tests/` that will be executed automatically by the CI workflow to check training, inference and API logic on every change.
- **CD**: An SSH-based deployment job in the same GitHub Actions workflow that, after a successful CI run on the `master` branch, connects to our Docker VM, pulls the latest code and restarts the stack with `docker compose up -d --build`.

After writing this plan, my next step will be to finish and refine the GitHub Actions workflow and start adding the first tests so that this CI/CT/CD pipeline actually runs end-to-end for the project.