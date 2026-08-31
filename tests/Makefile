# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
#
# Run from integration-testing/ (or set MOSAIC_PATH and use -f integration-testing/docker-compose.yml from repo root).
MOSAIC_PATH ?= $(abspath ..)
export MOSAIC_PATH

RESULTS_DIR ?= $(HOME)/tmp/mosaic
TESTS_DIR ?= $(MOSAIC_PATH)/tests/suites
TEST_MARKER ?= "profiler_otel or dashboards"

# Hardware profile to run against; see docs/docs/profiles.md ("Hardware Profiles").
# The profile also decides which stack setup/cleanup deploy. Other tiers live with the
# environments that run them and supply PROFILE_DIR.
PROFILE ?= default
# Extra directories of profile files, space separated.
PROFILE_DIR ?=
PROFILE_DIR_ARGS := $(foreach d,$(PROFILE_DIR),--profile-dir $(d))
PROFILE_ARGS := --workload-profile $(PROFILE) $(PROFILE_DIR_ARGS)

# The profile is what says which stack to deploy, so setup/cleanup ask it rather than naming
# a compose file. `compose-file` prints an absolute path, or nothing at all with a zero exit
# status when the profile is externally deployed. See docs/docs/profiles.md.
PROFILES_CLI = cd $(TESTS_DIR) && uv run python -m profiler_otel.profiles

FRAMEWORK_SERVICE_NAME ?= production-test-framework
FRAMEWORK_CONTAINER_NAME ?= production-test-framework

UV_INSTALL_URL ?= https://docs.astral.sh/uv/getting-started/installation/

.PHONY: check-uv setup run-tests copy-test-results test cleanup

# These targets should not stop the make workflow if they fail
.IGNORE: copy-test-results

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo ""; \
		echo "********************************************************************************"; \
		echo "Error: uv is not installed or not in the PATH."; \
		echo "You can install uv from:"; \
		echo "$(UV_INSTALL_URL)"; \
		echo "********************************************************************************"; \
		echo ""; \
		exit 1; \
	}

# Start the container stack the selected profile describes. An externally deployed profile
# expects the serving and observability stacks to be up already, so there is nothing to do.
setup: check-uv
	@set -e; \
	VLLM_COMPOSE="$$($(PROFILES_CLI) compose-file $(PROFILE) $(PROFILE_DIR_ARGS))"; \
	if [ -z "$$VLLM_COMPOSE" ]; then \
		echo "Profile '$(PROFILE)' is externally deployed; nothing to bring up."; \
		exit 0; \
	fi; \
	echo "Starting container stack for profile '$(PROFILE)'..."; \
	set -a; eval "$$($(PROFILES_CLI) deployment-env $(PROFILE) $(PROFILE_DIR_ARGS))"; set +a; \
	docker compose -f $(MOSAIC_PATH)/deployments/docker-compose.yml -f "$$VLLM_COMPOSE" \
		up --wait -d --pull always

cleanup: check-uv
	@set -e; \
	VLLM_COMPOSE="$$($(PROFILES_CLI) compose-file $(PROFILE) $(PROFILE_DIR_ARGS))"; \
	if [ -z "$$VLLM_COMPOSE" ]; then \
		echo "Profile '$(PROFILE)' is externally deployed; nothing to tear down."; \
		exit 0; \
	fi; \
	echo "Cleaning up container stack for profile '$(PROFILE)'..."; \
	docker compose -f $(MOSAIC_PATH)/deployments/docker-compose.yml -f "$$VLLM_COMPOSE" \
		down --remove-orphans \
	&& docker container prune -f

# Run the tests and exit with the pytest exit code
run-tests: check-uv
	@echo "$(TASK) Running tests with marker: $(TEST_MARKER), profile: $(PROFILE)..."; \
	cd $(TESTS_DIR) && \
	uv run pytest -m $(TEST_MARKER) $(PROFILE_ARGS) \
		--junitxml=$(RESULTS_DIR)/test_results.xml -v .; \
	exit $$?;

# Start the container stack, run the tests, and clean up
test: check-uv setup
	@$(MAKE) run-tests; EXIT_CODE=$$?; echo "EXIT_CODE: $$EXIT_CODE"; \
	$(MAKE) cleanup; \
	exit $$EXIT_CODE

