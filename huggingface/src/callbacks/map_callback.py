import pytorch_lightning as pl

class MAPCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        # 에포크 마지막에서 mAP 계산
        map_value = pl_module.train_map_metric.compute()
        
        # pl_module.log로 메트릭 기록
        pl_module.log("train_mAP", map_value['map'], logger=True)
        pl_module.log("train_mAP50", map_value['map_50'], prog_bar=True, logger=True)
        pl_module.log("train_mAP75", map_value['map_75'], logger=True)
        
        pl_module.train_map_metric.reset()  # 다음 에포크를 위해 metric 초기화

    def on_validation_epoch_end(self, trainer, pl_module):
        # 에포크 마지막에서 mAP 계산
        map_value = pl_module.val_map_metric.compute()
        
        # pl_module.log로 메트릭 기록
        pl_module.log("validation_mAP", map_value['map'], logger=True)
        pl_module.log("validation_mAP50", map_value['map_50'], prog_bar=True, logger=True)
        pl_module.log("validation_mAP75", map_value['map_75'], logger=True)
        
        pl_module.val_map_metric.reset()  # 다음 에포크를 위해 metric 초기화
