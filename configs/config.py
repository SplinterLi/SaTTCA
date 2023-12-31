# preprocessing configs
in_res = (128, 128, 128)
out_res = (16, 16, 16)
win_min, win_max = -1350, 150
model_path = "outputs/checkpoints/model.pth"             # The path of your pre-trained model weights 
csv_path = "/docs/set.csv"  # The path of set dividing .csv  
log_path = "/logs/"               # The path of logs
data_base = "/roi_files/"   # The path of database
dataset_list = [1, 2]

# training configs
batch_size = 2
num_workers = 4
max_lr = 1e-3
min_lr = 1e-6
epochs = 60
smooth = 1e-6
weight_decay = 1e-5

# testing configs
batch_size_test = 1
test_epoch = 10
test_batch_size = 1
test_lr = 1e-5
threshold = 1
diameter_rate = [0.8, 0.02]
test_set = (0, 1)

size_code = {
    1: "micro", # 0 ~ 10mm
    2: "small", # 10 ~ 20mm
    3: "medium",  # 20 ~ 30mm
    4: "mess"  # 30 ~ 64mm
}
