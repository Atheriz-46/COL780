python3 main.py -i data/ptz/input -o data/ptz/output -e data/ptz/temporalROI.txt -c b
python3 eval.py -g data/ptz/groundtruth -p data/ptz/output
python3 main.py -i data/ptz/input -o data/ptz/output -e data/ptz/temporalROI.txt -c i
python3 eval.py -g data/ptz/groundtruth -p data/ptz/output
python3 main.py -i data/ptz/input -o data/ptz/output -e data/ptz/temporalROI.txt -c j
python3 eval.py -g data/ptz/groundtruth -p data/ptz/output
python3 main.py -i data/ptz/input -o data/ptz/output -e data/ptz/temporalROI.txt -c m
python3 eval.py -g data/ptz/groundtruth -p data/ptz/output
0942
1052
0364
0528