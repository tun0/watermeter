DOCKER_UID := $(shell id -u)
DOCKER_GID := $(shell id -g)
COMPOSE = DOCKER_UID=$(DOCKER_UID) DOCKER_GID=$(DOCKER_GID) docker compose

.PHONY: reader gallery up down logs logs-gallery test

reader:
	$(COMPOSE) up -d --build meter_reader

gallery:
	$(COMPOSE) up -d --build gallery

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f meter_reader

logs-gallery:
	$(COMPOSE) logs -f gallery

test:
	$(COMPOSE) run --rm test
