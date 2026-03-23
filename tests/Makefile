# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
#
# Run from integration-testing/ (or set MOSAIC_PATH and use -f integration-testing/docker-compose.yml from repo root).
MOSAIC_PATH ?= $(abspath ..)
export MOSAIC_PATH

RESULTS_DIR ?= $(HOME)/tmp/mosaic

FRAMEWORK_SERVICE_NAME ?= production-test-framework
FRAMEWORK_CONTAINER_NAME ?= production-test-framework

.PHONY: setup run-tests copy-test-results test cleanup

# These targets should not stop the make workflow if they fail
.IGNORE: copy-test-results

setup:
	@echo "Starting container stack..."
	@docker compose --profile="*" pull && \
	docker compose up -d

cleanup:
	@echo "Cleaning up container stack..."
	@docker compose --profile "*" down --remove-orphans \
	&& docker container prune -f

run-tests:
	docker compose run --name $(FRAMEWORK_CONTAINER_NAME) $(FRAMEWORK_SERVICE_NAME) 

copy-test-results:
	rm -rf $(RESULTS_DIR)
	mkdir -p $(RESULTS_DIR) \
	&& docker cp $(FRAMEWORK_CONTAINER_NAME):/app/results/test_results.xml $(RESULTS_DIR)/test_results.xml

# Run the tests and copy the results to the results directory
test: setup
	@$(MAKE) run-tests; EXIT_CODE=$$?; echo "EXIT_CODE: $$EXIT_CODE"; \
	$(MAKE) copy-test-results; \
	$(MAKE) cleanup; \
	exit $$EXIT_CODE

