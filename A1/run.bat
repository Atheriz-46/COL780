py main.py -i test\test\test-ptz\input -o test\test\test-ptz\output -e test\test\test-ptz\eval_frames.txt -c b
py eval.py -g test\test\test-ptz\groundtruth -p test\test\test-ptz\
py main.py -i test\test\test-ptz\input -o test\test\test-ptz\output -e test\test\test-ptz\eval_frames.txt -c i
py eval.py -g test\test\test-ptz\groundtruth -p test\test\test-ptz\output
py main.py -i test\test\test-ptz\input -o test\test\test-ptz\output -e test\test\test-ptz\eval_frames.txt -c j
py eval.py -g test\test\test-ptz\groundtruth -p test\test\test-ptz\output
py main.py -i test\test\test-ptz\input -o test\test\test-ptz\output -e test\test\test-ptz\eval_frames.txt -c m
py eval.py -g test\test\test-ptz\groundtruth -p test\test\test-ptz\output

