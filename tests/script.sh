logger_dir=logs

mkdir -p logs

for dp_size in 1
do
    START_TIME=`date +%Y%m%d-%H:%M:%S`

    LOG_FILE=${logger_dir}/${START_TIME}-dp-size-${dp_size}.log
    torchrun --nproc_per_node 2 \
    test.py --dp-size ${dp_size} \
    2>&1 | tee -a $LOG_FILE
done
