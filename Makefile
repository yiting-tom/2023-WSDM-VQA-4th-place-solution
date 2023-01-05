docker_build:
	docker build . --tag wsdm2023 --network host

# docker run --rm -it --gpus all --network host -v /home/P76104419/wsdm/data:/mnt/data -v /home/P76104419/wsdm/output:/mnt/output wsdm2023
# docker run --rm -it --gpus all --network host -v /home/P76104419/wsdm/reproduce_vqa:/wsdm -v /home/P76104419/wsdm/data:/mnt/data -v /home/P76104419/wsdm/output:/mnt/output wsdm2023
docker_run:
	docker run --rm -it --gpus all --network host -v /home/P76104419/wsdm/reproduce_vqa:/wsdm -v /home/P76104419/wsdm/data:/mnt/data -v /home/P76104419/wsdm/output:/mnt/output wsdm2023