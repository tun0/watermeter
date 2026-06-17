DOCKER_UID := $(shell id -u)
DOCKER_GID := $(shell id -g)
COMPOSE = DOCKER_UID=$(DOCKER_UID) DOCKER_GID=$(DOCKER_GID) docker compose

.PHONY: collector up down logs test

collector:
	$(COMPOSE) up -d --build collector

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f collector

test:
	$(COMPOSE) run --rm test
