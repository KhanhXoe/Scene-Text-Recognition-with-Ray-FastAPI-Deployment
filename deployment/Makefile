.PHONY: init deploy client clean streamlit check deploy_ocr

clean:
	ray stop || true
	rm -rf /tmp/ray_project/*

init: clean
	mkdir -p /tmp/ray_project
	ray start --head --disable-usage-stats --node-ip-address=0.0.0.0 --dashboard-host=0.0.0.0 --temp-dir=/tmp/ray_project

check:
	@echo "where ray is installed:"
	which ray

deploy:
	@echo "Starting object detection server..."
	serve run testing_ray_server:entrypoint --address="0.0.0.0:6379"

deploy_ocr:
	@echo "Starting OCR server..."
	PYTHONPATH=.. serve run ocr:entrypoint --address="0.0.0.0:8000"

streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app.py --server.port=8051 --server.address=0.0.0.0
