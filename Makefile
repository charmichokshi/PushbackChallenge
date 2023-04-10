extract-data:
	@echo "Extracting data from the raw data files"
	find data -name 'K*.tar' -exec tar xvf {} -C data \;
	@echo "Deleting the raw data files"
	find data -name 'K*.tar' -exec rm {} \;

run-all-airports:
	@echo "Running benchmark on KATL"
	python3 benchmark_sequential.py --airport KATL &
	@echo "Running benchmark on KCLT"
	python3 benchmark_sequential.py --airport KCLT &
	@echo "Running benchmark on KDEN"
	python3 benchmark_sequential.py --airport KDEN &
	@echo "Running benchmark on KDFW"
	python3 benchmark_sequential.py --airport KDFW &
	@echo "Running benchmark on KJFK"
	python3 benchmark_sequential.py --airport KJFK &
	@echo "Running benchmark on KMEM"
	python3 benchmark_sequential.py --airport KMEM &
	@echo "Running benchmark on KMIA"
	python3 benchmark_sequential.py --airport KMIA &
	@echo "Running benchmark on KORD"
	python3 benchmark_sequential.py --airport KORD &
	@echo "Running benchmark on KPHX"
	python3 benchmark_sequential.py --airport KPHX &
	@echo "Running benchmark on KSEA"
	python3 benchmark_sequential.py --airport KSEA &


combine-all:
	@echo "Combining all the results"
	python benchmark_sequential.py
