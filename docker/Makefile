REGISTORY := registry.ritc.jp/ricos/simulation/femio
GHCR_IMAGE ?= ghcr.io/ricosjp/femio
TAG ?= 0.3.2

all: fistr

login:
	docker login $(REGISTORY)

fistr: Dockerfile
	cp ../pyproject.toml .
	docker build -t $(REGISTORY)/fistr_ricos:$(TAG) -f Dockerfile .

in: fistr
	docker run -it --rm $(REGISTORY)/fistr_ricos:$(TAG) /bin/bash

push: fistr
	docker push $(REGISTORY)/fistr_ricos:$(TAG)

ghcr: ghcr_login fistr
	docker build -t $(GHCR_IMAGE):$(TAG) \
		-f Dockerfile --build-arg ricos_pypi_key=$(key) .
	docker push $(GHCR_IMAGE):$(TAG)

ghcr_login:
	docker login $(GHCR_IMAGE)
