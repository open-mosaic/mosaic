# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
#
# Run from integration-testing/ (or set MOSAIC_PATH and use -f integration-testing/docker-compose.yml from repo root).
MOSAIC_PATH ?= $(abspath ..)
export MOSAIC_PATH

RESULTS_DIR ?= $(HOME)/tmp/mosaic
TESTS_DIR ?= $(MOSAIC_PATH)/tests/suites
TEST_MARKER ?= "profiler_otel or dashboards"

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

# Start the container stack
setup:
	@echo "Starting container stack..."
	docker compose -f $(MOSAIC_PATH)/deployments/docker-compose.yml -f $(MOSAIC_PATH)/deployments/docker-compose-vllm.yml up --wait -d --pull always

cleanup:
	@echo "Cleaning up container stack..."
	@docker compose -f $(MOSAIC_PATH)/deployments/docker-compose.yml -f $(MOSAIC_PATH)/deployments/docker-compose-vllm.yml down --remove-orphans \
	&& docker container prune -f

# Run the tests and exit with the pytest exit code
run-tests: check-uv
	@echo "$(TASK) Running tests with marker: $(TEST_MARKER)..."; \
	cd $(TESTS_DIR) && \
	uv run pytest -m $(TEST_MARKER) --junitxml=$(RESULTS_DIR)/test_results.xml -v .; \
	exit $$?;

# Start the container stack, run the tests, and clean up
test: check-uv setup
	@$(MAKE) run-tests; EXIT_CODE=$$?; echo "EXIT_CODE: $$EXIT_CODE"; \
	$(MAKE) cleanup; \
	exit $$EXIT_CODE

