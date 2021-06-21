REGISTORY := registry.ritc.jp/ricos/simulation/femio
TAG ?= 0.3.2

all: fistr

login:
	docker login $(REGISTORY)

fistr: Dockerfile
	cp ../pyproject.toml .
	docker build -t $(REGISTORY)/fistr_ricos:$(TAG) -f Dockerfile .

push: fistr
	docker push $(REGISTORY)/fistr_ricos:$(TAG)
