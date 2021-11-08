py eval_hog_pretrained.py --root .\ --test PennFudanPed_val.json --out output_1.json
py eval_detections.py --gt PennFudanPed_val.json --pred output_1.json